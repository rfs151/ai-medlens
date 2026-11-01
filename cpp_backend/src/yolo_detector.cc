#include "yolo_detector.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <json/json.h>
#include <opencv2/imgproc.hpp>

namespace medlens {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static double msSince(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

ModelMeta loadMeta(const std::string& onnx_path) {
    ModelMeta meta;
    fs::path p(onnx_path);
    meta.name = p.stem().string();
    fs::path sidecar = p.replace_extension(".json");
    std::ifstream in(sidecar);
    if (in) {
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errs;
        if (Json::parseFromStream(builder, in, &root, &errs)) {
            for (const auto& n : root["names"]) meta.names.push_back(n.asString());
            meta.imgsz = root.get("imgsz", 640).asInt();
            meta.description = root.get("description", "").asString();
            meta.license = root.get("license", "").asString();
            if (root.isMember("recommended_conf")) meta.recommended_conf = root["recommended_conf"].asDouble();
            meta.extra = Json::objectValue;
            for (const char* key : {"metrics", "recommended_iou", "count_groups", "max_box_area", "source"})
                if (root.isMember(key)) meta.extra[key] = root[key];
        }
    }
    return meta;
}

YoloDetector::YoloDetector(const std::string& onnx_path, const ModelMeta& meta, bool use_cuda)
    : net_(cv::dnn::readNetFromONNX(onnx_path)), meta_(meta) {
    if (use_cuda) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& bgr, float conf_thr, float iou_thr, Timing* timing) {
    auto t0 = Clock::now();
    const int sz = meta_.imgsz;

    // ---- letterbox (same math as Ultralytics / the Python backend) ----
    const float scale = std::min(sz / float(bgr.rows), sz / float(bgr.cols));
    const int nw = int(std::round(bgr.cols * scale)), nh = int(std::round(bgr.rows * scale));
    const float pad_x = (sz - nw) / 2.0f, pad_y = (sz - nh) / 2.0f;
    const int left = int(std::round(pad_x - 0.1f)), top = int(std::round(pad_y - 0.1f));
    cv::Mat resized, padded;
    cv::resize(bgr, resized, {nw, nh}, 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, padded, top, sz - nh - top, left, sz - nw - left, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, {sz, sz}, cv::Scalar(), /*swapRB=*/true, false);
    auto t1 = Clock::now();

    net_.setInput(blob);
    cv::Mat out = net_.forward();
    auto t2 = Clock::now();

    // ---- decode: (1, 4+nc, N) for v8/11, (1, N, 5+nc) for v5 ----
    const int nc = int(meta_.names.size());
    cv::Mat pred(out.size[1], out.size[2], CV_32F, out.ptr<float>());
    bool v5 = false;
    if (pred.rows == 4 + nc && pred.cols != 4 + nc) {
        pred = pred.t();  // -> (N, 4+nc)
    } else if (pred.cols == 5 + nc) {
        v5 = true;
    } else if (pred.cols != 4 + nc) {
        throw std::runtime_error("unexpected output shape for " + std::to_string(nc) + " classes");
    }

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    const int cls_off = v5 ? 5 : 4;
    for (int i = 0; i < pred.rows; ++i) {
        const float* r = pred.ptr<float>(i);
        const float* best = std::max_element(r + cls_off, r + cls_off + nc);
        float score = *best * (v5 ? r[4] : 1.0f);
        if (score < conf_thr) continue;
        boxes.emplace_back(r[0] - r[2] / 2, r[1] - r[3] / 2, r[2], r[3]);
        scores.push_back(score);
        class_ids.push_back(int(best - (r + cls_off)));
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxesBatched(boxes, scores, class_ids, conf_thr, iou_thr, keep);

    std::vector<Detection> dets;
    dets.reserve(keep.size());
    const float W = float(bgr.cols), H = float(bgr.rows);
    for (int k : keep) {
        const auto& b = boxes[k];
        auto ux = [&](double v) { return std::clamp(float((v - left) / scale), 0.0f, W); };
        auto uy = [&](double v) { return std::clamp(float((v - top) / scale), 0.0f, H); };
        int c = class_ids[k];
        dets.push_back({c, c < nc ? meta_.names[c] : std::to_string(c), scores[k], ux(b.x), uy(b.y),
                        ux(b.x + b.width), uy(b.y + b.height)});
    }
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.conf > b.conf; });

    if (timing) {
        timing->preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timing->inference_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        timing->postprocess_ms = msSince(t2);
        timing->total_ms = msSince(t0);
    }
    return dets;
}

DetectorPool::DetectorPool(const std::string& onnx_path, const ModelMeta& meta, int size, bool use_cuda)
    : meta_(meta), use_cuda_(use_cuda) {
    // The first forward() of a cv::dnn::Net allocates and fuses layers (several times slower),
    // so run one dummy frame per instance before serving.
    const cv::Mat dummy(meta.imgsz, meta.imgsz, CV_8UC3, cv::Scalar(114, 114, 114));
    for (int i = 0; i < std::max(1, size); ++i) {
        auto det = std::make_unique<YoloDetector>(onnx_path, meta, use_cuda);
        det->detect(dummy, 0.99f, 0.5f);
        idle_.push_back(std::move(det));
    }
}

std::vector<Detection> DetectorPool::detect(const cv::Mat& bgr, float conf, float iou, Timing* timing) {
    std::unique_ptr<YoloDetector> det;
    {
        std::unique_lock lk(mu_);
        cv_.wait(lk, [&] { return !idle_.empty(); });
        det = std::move(idle_.back());
        idle_.pop_back();
    }
    struct Return {  // give the detector back even if detect() throws
        DetectorPool* pool;
        std::unique_ptr<YoloDetector>& d;
        ~Return() {
            {
                std::lock_guard lk(pool->mu_);
                pool->idle_.push_back(std::move(d));
            }
            pool->cv_.notify_one();
        }
    } guard{this, det};
    return det->detect(bgr, conf, iou, timing);
}

}  // namespace medlens
