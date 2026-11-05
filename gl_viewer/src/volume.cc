#include "volume.h"

#include <algorithm>
#include <cstring>

namespace mv {
namespace {

constexpr size_t kHeaderSize = 128;

template <typename T>
T readLE(const uint8_t* p) {  // file is little endian, like every platform we target
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

// In-place separable [1 2 1]/4 blur of one channel of an interleaved RGBA8 volume.
void blur121(std::vector<uint8_t>& rgba, int channel, const std::array<int, 3>& d) {
    const size_t strides[3] = {1, size_t(d[0]), size_t(d[0]) * d[1]};
    std::vector<uint16_t> line;
    for (int axis = 0; axis < 3; ++axis) {
        const int n = d[axis];
        const size_t step = strides[axis] * 4;
        line.resize(size_t(n));
        // iterate over every line parallel to `axis`
        const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
        for (int j = 0; j < d[a2]; ++j)
            for (int i = 0; i < d[a1]; ++i) {
                uint8_t* base = rgba.data() + (size_t(i) * strides[a1] + size_t(j) * strides[a2]) * 4 + channel;
                for (int k = 0; k < n; ++k) line[size_t(k)] = base[size_t(k) * step];
                for (int k = 0; k < n; ++k) {
                    int prev = line[size_t(std::max(k - 1, 0))], next = line[size_t(std::min(k + 1, n - 1))];
                    base[size_t(k) * step] = uint8_t((prev + 2 * line[size_t(k)] + next + 2) / 4);
                }
            }
    }
}

}  // namespace

bool Volume::parse(const uint8_t* data, size_t size, std::string* error) {
    auto fail = [&](const char* msg) {
        if (error) *error = msg;
        return false;
    };
    if (size < kHeaderSize || std::memcmp(data, "MLVOL01", 8) != 0) return fail("not an .mlvol file");
    for (int i = 0; i < 3; ++i) {
        dims[size_t(i)] = readLE<int32_t>(data + 8 + 4 * i);
        spacing[size_t(i)] = readLE<float>(data + 20 + 4 * i);
    }
    const int nch = readLE<int32_t>(data + 32);
    const bool hasLabels = readLE<int32_t>(data + 36) != 0;
    if (dims[0] < 2 || dims[1] < 2 || dims[2] < 2 || dims[0] > 2048 || dims[1] > 2048 || dims[2] > 2048)
        return fail("bad dimensions");
    if (nch < 1 || nch > 4) return fail("bad channel count");
    const size_t n = voxelCount();
    if (size < kHeaderSize + n * size_t(nch + (hasLabels ? 1 : 0))) return fail("file truncated");

    channelNames.clear();
    channels.clear();
    const uint8_t* p = data + kHeaderSize;
    for (int c = 0; c < nch; ++c) {
        const char* name = reinterpret_cast<const char*>(data + 40 + 16 * c);
        channelNames.emplace_back(name, strnlen(name, 16));
        channels.emplace_back(p, p + n);
        p += n;
    }
    labels = hasLabels ? std::vector<uint8_t>(p, p + n) : std::vector<uint8_t>{};
    return true;
}

std::vector<uint8_t> Volume::buildTumourMasks() const {
    const size_t n = voxelCount();
    std::vector<uint8_t> rgba(n * 4, 0);
    if (labels.empty()) return rgba;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t l = labels[i];
        if (l >= 1 && l <= kTumourClasses) rgba[i * 4 + (l - 1)] = 255;
    }
    for (int c = 0; c < kTumourClasses; ++c) blur121(rgba, c, dims);
    return rgba;
}

std::array<size_t, 4> Volume::labelCounts() const {
    std::array<size_t, 4> counts{0, 0, 0, 0};
    for (uint8_t l : labels)
        if (l < 4) ++counts[l];
    return counts;
}

}  // namespace mv
