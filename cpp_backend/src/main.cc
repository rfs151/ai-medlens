// MedLens C++ backend: same /api/detect contract as the Python service, so the web page
// can switch backends and compare latency side by side.
//
//   GET  /api/health
//   GET  /api/models
//   POST /api/detect?model=&conf=&iou=   multipart field "file" (JPG/PNG)
//
// Env: MEDLENS_MODELS_DIR (default ../models), MEDLENS_PORT (8081),
//      MEDLENS_THREADS (HTTP threads, default = cores), MEDLENS_POOL_SIZE (nets per model, default = threads),
//      MEDLENS_CUDA=1 (needs OpenCV built with CUDA)

#include <drogon/drogon.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "yolo_detector.h"

using namespace drogon;
namespace fs = std::filesystem;

static std::map<std::string, std::shared_ptr<medlens::DetectorPool>> g_models;

static std::string envOr(const char* key, const std::string& def) {
    const char* v = std::getenv(key);
    return v && *v ? v : def;
}

static HttpResponsePtr jsonError(HttpStatusCode code, const std::string& msg) {
    Json::Value v;
    v["detail"] = msg;
    auto resp = HttpResponse::newHttpJsonResponse(v);
    resp->setStatusCode(code);
    return resp;
}

static std::shared_ptr<medlens::DetectorPool> findModel(const std::string& name) {
    if (g_models.empty()) return nullptr;
    if (name.empty()) return g_models.begin()->second;
    auto it = g_models.find(name);
    return it == g_models.end() ? nullptr : it->second;
}

static float paramOr(const HttpRequestPtr& req, const std::string& key, float def) {
    const auto& s = req->getParameter(key);
    try {
        return s.empty() ? def : std::stof(s);
    } catch (...) {
        return def;
    }
}

int main() {
    const std::string models_dir = envOr("MEDLENS_MODELS_DIR", "../models");
    const int port = std::stoi(envOr("MEDLENS_PORT", "8081"));
    const int threads = std::stoi(envOr("MEDLENS_THREADS", std::to_string(std::max(1u, std::thread::hardware_concurrency()))));
    const int pool_size = std::stoi(envOr("MEDLENS_POOL_SIZE", std::to_string(threads)));
    const bool use_cuda = envOr("MEDLENS_CUDA", "0") == "1";

    if (fs::exists(models_dir)) {
        for (const auto& entry : fs::directory_iterator(models_dir)) {
            if (entry.path().extension() != ".onnx") continue;
            try {
                auto meta = medlens::loadMeta(entry.path().string());
                if (meta.names.empty()) {
                    LOG_WARN << "skip " << entry.path() << ": missing <name>.json with class names";
                    continue;
                }
                g_models[meta.name] =
                    std::make_shared<medlens::DetectorPool>(entry.path().string(), meta, pool_size, use_cuda);
                LOG_INFO << "loaded model " << meta.name << " (" << meta.names.size() << " classes, pool "
                         << pool_size << ")";
            } catch (const std::exception& e) {
                LOG_ERROR << "failed to load " << entry.path() << ": " << e.what();
            }
        }
    } else {
        LOG_WARN << "models dir not found: " << models_dir;
    }

    app().registerHandler("/api/health", [](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
        Json::Value v;
        v["status"] = "ok";
        v["backend"] = "cpp-drogon-opencv-dnn";
        v["models"] = Json::UInt(g_models.size());
        cb(HttpResponse::newHttpJsonResponse(v));
    }, {Get});

    app().registerHandler("/api/models", [](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
        Json::Value v;
        v["backend"] = "cpp-drogon-opencv-dnn";
        v["models"] = Json::arrayValue;
        for (const auto& [name, pool] : g_models) {
            const auto& m = pool->meta();
            Json::Value j;
            j["name"] = name;
            j["imgsz"] = m.imgsz;
            j["provider"] = pool->provider();
            j["description"] = m.description;
            j["license"] = m.license;
            j["classes"] = Json::arrayValue;
            for (const auto& c : m.names) j["classes"].append(c);
            if (m.recommended_conf >= 0) j["recommended_conf"] = m.recommended_conf;
            for (const auto& key : m.extra.getMemberNames()) j[key] = m.extra[key];
            v["models"].append(j);
        }
        cb(HttpResponse::newHttpJsonResponse(v));
    }, {Get});

    app().registerHandler("/api/detect", [](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto model = findModel(req->getParameter("model"));
        if (!model) return cb(jsonError(k404NotFound, "model not found"));

        MultiPartParser parser;
        if (parser.parse(req) != 0 || parser.getFiles().empty())
            return cb(jsonError(k400BadRequest, "multipart field 'file' required"));
        const auto& file = parser.getFiles()[0];

        auto t0 = std::chrono::steady_clock::now();
        cv::Mat buf(1, int(file.fileLength()), CV_8U, const_cast<char*>(file.fileData()));
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
        double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (img.empty()) return cb(jsonError(k400BadRequest, "unsupported image format (DICOM: use the Python backend)"));

        medlens::Timing t;
        std::vector<medlens::Detection> dets;
        try {
            dets = model->detect(img, paramOr(req, "conf", 0.25f), paramOr(req, "iou", 0.45f), &t);
        } catch (const std::exception& e) {
            return cb(jsonError(k500InternalServerError, e.what()));
        }

        Json::Value v;
        v["backend"] = "cpp-drogon-opencv-dnn";
        v["model"] = model->meta().name;
        v["provider"] = model->provider();
        v["image"]["width"] = img.cols;
        v["image"]["height"] = img.rows;
        v["boxes"] = Json::arrayValue;
        Json::Value counts(Json::objectValue);
        for (const auto& d : dets) {
            Json::Value b;
            b["cls"] = d.cls;
            b["label"] = d.label;
            b["conf"] = std::round(d.conf * 10000) / 10000;
            b["x1"] = d.x1; b["y1"] = d.y1; b["x2"] = d.x2; b["y2"] = d.y2;
            v["boxes"].append(b);
            counts[d.label] = counts.get(d.label, 0).asInt() + 1;
        }
        v["counts"] = counts;
        auto r2 = [](double x) { return std::round(x * 100) / 100; };
        v["timing_ms"]["decode"] = r2(decode_ms);
        v["timing_ms"]["preprocess"] = r2(t.preprocess_ms);
        v["timing_ms"]["inference"] = r2(t.inference_ms);
        v["timing_ms"]["postprocess"] = r2(t.postprocess_ms);
        v["timing_ms"]["total"] = r2(t.total_ms);
        cb(HttpResponse::newHttpJsonResponse(v));
    }, {Post});

    // CORS: the page is served by the Python service, so calls here are cross-origin.
    app().registerPreRoutingAdvice([](const HttpRequestPtr& req, AdviceCallback&& acb, AdviceChainCallback&& ccb) {
        if (req->method() == Options) {
            auto resp = HttpResponse::newHttpResponse();
            resp->addHeader("Access-Control-Allow-Origin", "*");
            resp->addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            resp->addHeader("Access-Control-Allow-Headers", "*");
            return acb(resp);
        }
        ccb();
    });
    app().registerPostHandlingAdvice([](const HttpRequestPtr&, const HttpResponsePtr& resp) {
        resp->addHeader("Access-Control-Allow-Origin", "*");
    });

    LOG_INFO << "MedLens C++ backend on :" << port << " with " << g_models.size() << " model(s)";
    app().setClientMaxBodySize(30 * 1024 * 1024)
        .setThreadNum(threads)
        .addListener("0.0.0.0", port)
        .run();
}
