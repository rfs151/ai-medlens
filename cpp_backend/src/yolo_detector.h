#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <json/json.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace medlens {

struct Detection {
    int cls;
    std::string label;
    float conf;
    float x1, y1, x2, y2;
};

struct Timing {
    double preprocess_ms = 0, inference_ms = 0, postprocess_ms = 0, total_ms = 0;
};

struct ModelMeta {
    std::string name;
    std::vector<std::string> names;
    int imgsz = 640;
    std::string description;
    std::string license;
    double recommended_conf = -1;
    Json::Value extra;  // display-only fields passed through to /api/models (metrics, source, max_box_area, ...)
};

// One cv::dnn::Net instance. Not thread-safe: use DetectorPool for concurrent requests.
class YoloDetector {
public:
    YoloDetector(const std::string& onnx_path, const ModelMeta& meta, bool use_cuda);
    std::vector<Detection> detect(const cv::Mat& bgr, float conf, float iou, Timing* timing = nullptr);

private:
    cv::dnn::Net net_;
    ModelMeta meta_;
};

// Fixed-size pool of detectors sharing one model file; borrow() blocks while all are busy.
class DetectorPool {
public:
    DetectorPool(const std::string& onnx_path, const ModelMeta& meta, int size, bool use_cuda);
    std::vector<Detection> detect(const cv::Mat& bgr, float conf, float iou, Timing* timing);
    const ModelMeta& meta() const { return meta_; }
    std::string provider() const { return use_cuda_ ? "opencv-dnn-cuda" : "opencv-dnn-cpu"; }

private:
    ModelMeta meta_;
    bool use_cuda_;
    std::vector<std::unique_ptr<YoloDetector>> idle_;
    std::mutex mu_;
    std::condition_variable cv_;
};

// Reads <name>.json written by training/export_onnx.py
ModelMeta loadMeta(const std::string& onnx_path);

}  // namespace medlens
