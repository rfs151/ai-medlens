// Pure inference benchmark (no HTTP): ./bench_detector ../models/polyp.onnx sample.jpg 200
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "yolo_detector.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.onnx image.jpg [runs]\n", argv[0]);
        return 1;
    }
    const int runs = argc > 3 ? std::atoi(argv[3]) : 100;
    auto meta = medlens::loadMeta(argv[1]);
    medlens::YoloDetector det(argv[1], meta, false);
    cv::Mat img = cv::imread(argv[2]);
    if (img.empty()) return 1;

    for (int i = 0; i < 5; ++i) det.detect(img, 0.25f, 0.45f);  // warm-up
    std::vector<double> total, infer;
    size_t n = 0;
    for (int i = 0; i < runs; ++i) {
        medlens::Timing t;
        n = det.detect(img, 0.25f, 0.45f, &t).size();
        total.push_back(t.total_ms);
        infer.push_back(t.inference_ms);
    }
    auto pct = [](std::vector<double> v, double p) {
        std::sort(v.begin(), v.end());
        return v[size_t(p * (v.size() - 1))];
    };
    std::printf("boxes=%zu runs=%d  total p50=%.2fms p95=%.2fms  inference p50=%.2fms  => %.1f FPS\n", n, runs,
                pct(total, 0.5), pct(total, 0.95), pct(infer, 0.5), 1000.0 / pct(total, 0.5));
    return 0;
}
