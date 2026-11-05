#pragma once
// CPU-side volume: 8-bit intensity channels + tumour labels, parsed from .mlvol (see scripts/fetch_volume.py).

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mv {

struct Volume {
    std::array<int, 3> dims{0, 0, 0};             // x (fastest), y, z
    std::array<float, 3> spacing{1, 1, 1};        // mm
    std::vector<std::string> channelNames;
    std::vector<std::vector<uint8_t>> channels;  // each dims[0]*dims[1]*dims[2]
    std::vector<uint8_t> labels;                  // empty if the file has none; 1..3 = tumour classes

    static constexpr int kTumourClasses = 3;

    size_t voxelCount() const { return size_t(dims[0]) * dims[1] * dims[2]; }
    // Physical size in mm along each axis.
    std::array<float, 3> extentMm() const {
        return {dims[0] * spacing[0], dims[1] * spacing[1], dims[2] * spacing[2]};
    }

    // Parses an uncompressed .mlvol buffer. Returns false with a message on malformed input.
    bool parse(const uint8_t* data, size_t size, std::string* error);

    // RGBA8 volume where R/G/B hold soft masks of label 1/2/3 (3x3x3 smoothed so trilinear
    // sampling gives smooth iso-surfaces instead of voxel staircases). A is unused.
    std::vector<uint8_t> buildTumourMasks() const;

    // Voxel count per label 0..3 and the matching volume in millilitres.
    std::array<size_t, 4> labelCounts() const;
    double voxelMl() const { return double(spacing[0]) * spacing[1] * spacing[2] / 1000.0; }
};

}  // namespace mv
