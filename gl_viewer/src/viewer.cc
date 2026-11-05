#include "viewer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "shaders.h"

namespace mv {
namespace {

constexpr double kRefineDelay = 0.25;  // seconds without interaction before rendering at full resolution
constexpr float kLowResScale = 0.5f;   // render scale of the 3D view while interacting
constexpr int kGap = 2;                // pixels between layout cells
constexpr float kFovY = 0.52f;         // ~30 degrees

// Slicer-style plane colours: axial red, coronal green, sagittal yellow.
const vec3 kPlaneColor[3] = {{0.95f, 0.35f, 0.35f}, {0.35f, 0.9f, 0.45f}, {0.98f, 0.82f, 0.3f}};
const vec3 kVolumeBorder{0.35f, 0.55f, 0.85f};
const vec3 kTumourAlpha{0.14f, 0.95f, 1.0f};  // edema translucent, cores nearly opaque
// Tumour classes: edema, non-enhancing core, enhancing core.
const float kTumourColor[9] = {0.98f, 0.82f, 0.25f, 0.95f, 0.45f, 0.15f, 0.9f, 0.12f, 0.2f};

void setTexParams(GLenum target, GLint filter) {
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (target == GL_TEXTURE_3D) glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}

Texture newTexture() {
    GLuint id = 0;
    glGenTextures(1, &id);
    return Texture(id);
}

float clamp01(float v) { return std::min(1.0f, std::max(0.0f, v)); }

}  // namespace

bool Viewer::init(std::string* log) {
    volumeProg_ = Program::build(shaders::kFullscreenVS, shaders::kVolumeFS, log);
    sliceProg_ = Program::build(shaders::kFullscreenVS, shaders::kSliceFS, log);
    blitProg_ = Program::build(shaders::kFullscreenVS, shaders::kBlitFS, log);
    if (!volumeProg_ || !sliceProg_ || !blitProg_) return false;

    GLuint vao = 0;  // core profiles need a bound VAO even though the triangle has no attributes
    glGenVertexArrays(1, &vao);
    vao_.reset(vao);

    tfTex_ = newTexture();
    glBindTexture(GL_TEXTURE_2D, tfTex_.get());
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    setTexParams(GL_TEXTURE_2D, GL_LINEAR);
    uploadTransferFunction();
    resetView();
    return true;
}

bool Viewer::setVolume(Volume volume, std::string* error) {
    if (volume.voxelCount() == 0 || volume.channels.empty()) {
        if (error) *error = "empty volume";
        return false;
    }
    vol_ = std::move(volume);
    const auto ext = vol_.extentMm();
    const float maxExt = std::max(ext[0], std::max(ext[1], ext[2]));
    boxHalf_ = vec3(ext[0], ext[1], ext[2]) * (0.5f / maxExt);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  // rows of an odd-width volume are not 4-byte aligned
    const std::vector<uint8_t> masks = vol_.buildTumourMasks();
    maskTex_ = newTexture();
    glBindTexture(GL_TEXTURE_3D, maskTex_.get());
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, vol_.dims[0], vol_.dims[1], vol_.dims[2], 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, masks.data());
    setTexParams(GL_TEXTURE_3D, GL_LINEAR);

    volTex_ = newTexture();
    uploadedChannel_ = -1;
    settings_.channel = std::min(settings_.channel, int(vol_.channels.size()) - 1);
    uploadChannel();

    // Start the crosshair on the tumour (centroid of the enhancing core, else of any label).
    cursor_ = vec3(0.5f, 0.5f, 0.5f);
    if (!vol_.labels.empty()) {
        for (int wanted : {3, -1}) {
            double sx = 0, sy = 0, sz = 0;
            size_t n = 0, idx = 0;
            for (int z = 0; z < vol_.dims[2]; ++z)
                for (int y = 0; y < vol_.dims[1]; ++y)
                    for (int x = 0; x < vol_.dims[0]; ++x, ++idx) {
                        const int l = vol_.labels[idx];
                        if (wanted < 0 ? l > 0 : l == wanted) {
                            sx += x; sy += y; sz += z; ++n;
                        }
                    }
            if (n) {
                cursor_ = vec3(float((sx / n + 0.5) / vol_.dims[0]), float((sy / n + 0.5) / vol_.dims[1]),
                               float((sz / n + 0.5) / vol_.dims[2]));
                break;
            }
        }
    }
    resetView();
    return true;
}

void Viewer::uploadChannel() {
    if (!hasVolume() || uploadedChannel_ == settings_.channel) return;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_3D, volTex_.get());
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, vol_.dims[0], vol_.dims[1], vol_.dims[2], 0, GL_RED, GL_UNSIGNED_BYTE,
                 vol_.channels[size_t(settings_.channel)].data());
    setTexParams(GL_TEXTURE_3D, GL_LINEAR);
    uploadedChannel_ = settings_.channel;
}

// 256-entry RGBA lookup table. Alpha is per voxel-sized step; the shader rescales it for its step.
void Viewer::uploadTransferFunction() {
    uint8_t lut[256 * 4];
    const float lo = settings_.windowLo, hi = std::max(settings_.windowHi, settings_.windowLo + 0.01f);
    for (int i = 0; i < 256; ++i) {
        const float x = i / 255.0f, t = clamp01((x - lo) / (hi - lo));
        float r, g, b, a;
        switch (settings_.preset) {
            case 0:  // grey cloud
                r = g = b = 0.3f + 0.7f * t;
                a = 0.06f * std::pow(t, 1.4f);
                break;
            case 2: {  // opaque boundary: shows the brain surface, use the cut plane to look inside
                const float s = t * t * (3 - 2 * t);
                r = 0.92f * (0.55f + 0.45f * t); g = 0.78f * (0.55f + 0.45f * t); b = 0.7f * (0.55f + 0.45f * t);
                a = 0.6f * clamp01((s - 0.12f) / 0.2f);
                break;
            }
            default:  // warm tissue ramp
                r = 0.55f + 0.45f * t; g = 0.25f + 0.65f * t; b = 0.18f + 0.6f * t * t;
                a = 0.07f * std::pow(t, 1.3f);
                break;
        }
        if (i < 2) a = 0;  // background outside the brain
        lut[i * 4 + 0] = uint8_t(clamp01(r) * 255);
        lut[i * 4 + 1] = uint8_t(clamp01(g) * 255);
        lut[i * 4 + 2] = uint8_t(clamp01(b) * 255);
        lut[i * 4 + 3] = uint8_t(clamp01(a) * 255 + 0.5f);
    }
    glBindTexture(GL_TEXTURE_2D, tfTex_.get());
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 1, GL_RGBA, GL_UNSIGNED_BYTE, lut);
}

void Viewer::resetView() {
    // Front-left view from slightly above (world: x right, y anterior, z superior).
    rot_ = (quat::axisAngle({0, 0, 1}, -0.6f) * quat::axisAngle({1, 0, 0}, 0.3f)).normalized();
    distance_ = 1.9f;
    volumeDirty_ = true;
}

void Viewer::resize(int width, int height) {
    width = std::max(width, 1);
    height = std::max(height, 1);
    if (width == width_ && height == height_) return;
    width_ = width;
    height_ = height;
    volumeDirty_ = slicesDirty_ = true;
}

Viewer::Rect Viewer::cellRect(int cell) const {
    if (settings_.layout == 1) return {0, 0, width_, height_};
    const int w0 = (width_ - kGap) / 2, h0 = (height_ - kGap) / 2;
    const int w1 = width_ - w0 - kGap, h1 = height_ - h0 - kGap;
    switch (cell) {
        case kAxial: return {0, 0, w0, h0};
        case kVolume: return {w0 + kGap, 0, w1, h0};
        case kCoronal: return {0, h0 + kGap, w0, h1};
        default: return {w0 + kGap, h0 + kGap, w1, h1};
    }
}

int Viewer::cellAt(float x, float y) const {
    for (int c = 0; c < 4; ++c) {
        if (!cellVisible(c)) continue;
        Rect r = cellRect(c);
        if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) return c;
    }
    return -1;
}

// Texture-space plane for each MPR view, oriented the radiological way (anterior up on axial,
// superior up on coronal/sagittal). The shader and picking share this one definition.
void Viewer::sliceBasis(int cell, vec3* origin, vec3* u, vec3* v, float* physW, float* physH) const {
    const auto ext = vol_.extentMm();
    switch (cell) {
        case kAxial:
            *origin = {0, 1, cursor_.z}; *u = {1, 0, 0}; *v = {0, -1, 0};
            *physW = ext[0]; *physH = ext[1];
            break;
        case kCoronal:
            *origin = {0, cursor_.y, 0}; *u = {1, 0, 0}; *v = {0, 0, 1};
            *physW = ext[0]; *physH = ext[2];
            break;
        default:
            *origin = {cursor_.x, 0, 0}; *u = {0, 1, 0}; *v = {0, 0, 1};
            *physW = ext[1]; *physH = ext[2];
            break;
    }
}

void Viewer::sliceFit(int cell, const Rect& r, float* fx, float* fy) const {
    vec3 o, u, v;
    float pw, ph;
    sliceBasis(cell, &o, &u, &v, &pw, &ph);
    const float cellAspect = float(r.w) / float(std::max(r.h, 1)), sliceAspect = pw / ph;
    *fx = cellAspect > sliceAspect ? cellAspect / sliceAspect : 1.0f;
    *fy = cellAspect > sliceAspect ? 1.0f : sliceAspect / cellAspect;
    *fx *= 1.04f;  // small margin around the slice
    *fy *= 1.04f;
}

void Viewer::pickSlice(int cell, float x, float y) {
    const Rect r = cellRect(cell);
    float fx, fy;
    sliceFit(cell, r, &fx, &fy);
    const float su = clamp01(((x - r.x) / r.w - 0.5f) * fx + 0.5f);
    const float sv = clamp01((1.0f - (y - r.y) / r.h - 0.5f) * fy + 0.5f);
    vec3 o, u, v;
    float pw, ph;
    sliceBasis(cell, &o, &u, &v, &pw, &ph);
    const vec3 p = o + u * su + v * sv;
    // keep the coordinate this view does not control
    if (cell == kAxial) cursor_ = {p.x, p.y, cursor_.z};
    else if (cell == kCoronal) cursor_ = {p.x, cursor_.y, p.z};
    else cursor_ = {cursor_.x, p.y, p.z};
    slicesDirty_ = true;
    if (settings_.showPlanes) volumeDirty_ = true;
}

void Viewer::pointerDown(float x, float y, int button, double now) {
    dragCell_ = cellAt(x, y);
    dragButton_ = button;
    lastX_ = x;
    lastY_ = y;
    if (dragCell_ >= 0 && dragCell_ != kVolume && hasVolume()) {
        pickSlice(dragCell_, x, y);
        if (settings_.showPlanes) touch(now);
    }
}

void Viewer::pointerMove(float x, float y, double now) {
    if (dragCell_ < 0) return;
    const float dx = x - lastX_, dy = y - lastY_;
    lastX_ = x;
    lastY_ = y;
    if (dragCell_ == kVolume) {
        // trackball: horizontal drag spins around the screen's up axis, vertical around its right axis
        const vec3 up = rot_.rotate({0, 0, 1}), back = rot_.rotate({0, 1, 0});
        const vec3 right = cross(up, back);
        const float k = 3.0f / float(std::max(height_, 200));
        rot_ = (quat::axisAngle(up, -dx * k) * quat::axisAngle(right, -dy * k) * rot_).normalized();
        volumeDirty_ = true;
        touch(now);
    } else if (hasVolume()) {
        pickSlice(dragCell_, x, y);
        if (settings_.showPlanes) touch(now);
    }
}

void Viewer::pointerUp(double now) {
    dragCell_ = -1;
    touch(now);
}

void Viewer::wheel(float x, float y, float delta, double now) {
    const int cell = cellAt(x, y);
    if (cell == kVolume) {
        distance_ = std::min(6.0f, std::max(0.9f, distance_ * std::pow(1.12f, -delta)));
        volumeDirty_ = true;
        touch(now);
    } else if (cell >= 0 && hasVolume()) {  // step one voxel through the slices
        const float s = delta > 0 ? 1.0f : -1.0f;
        if (cell == kAxial) cursor_.z = clamp01(cursor_.z + s / vol_.dims[2]);
        else if (cell == kCoronal) cursor_.y = clamp01(cursor_.y - s / vol_.dims[1]);
        else cursor_.x = clamp01(cursor_.x + s / vol_.dims[0]);
        slicesDirty_ = true;
        if (settings_.showPlanes) {
            volumeDirty_ = true;
            touch(now);
        }
    }
}

void Viewer::settingsChanged(double now) {
    if (hasVolume()) {
        settings_.channel = std::min(std::max(settings_.channel, 0), int(vol_.channels.size()) - 1);
        uploadChannel();
    }
    uploadTransferFunction();
    volumeDirty_ = slicesDirty_ = true;
    touch(now);
}

bool Viewer::needsRedraw(double now) const {
    return volumeDirty_ || slicesDirty_ || (lowResShown_ && now - lastInteraction_ >= kRefineDelay);
}

void Viewer::render(double now, GLuint target) {
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, width_, height_);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glClearColor(0.07f, 0.08f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(vao_.get());
    seed_ = std::fmod(seed_ + 0.618f, 1.0f);

    if (hasVolume()) {
        for (int c : {kAxial, kCoronal, kSagittal})
            if (cellVisible(c)) drawSlice(c, cellRect(c));

        const Rect r = cellRect(kVolume);
        const bool interacting = now - lastInteraction_ < kRefineDelay;
        const bool refine = lowResShown_ && !interacting;
        if (volumeDirty_ || refine || volTarget_.width() == 0) {
            const float scale = interacting ? kLowResScale : 1.0f;
            drawVolume(r, scale);
            lowResShown_ = interacting;
            glBindFramebuffer(GL_FRAMEBUFFER, target);
        }
        blitVolume(r);
    }
    volumeDirty_ = slicesDirty_ = false;
}

void Viewer::drawSlice(int cell, const Rect& r) {
    vec3 o, u, v;
    float pw, ph, fx, fy;
    sliceBasis(cell, &o, &u, &v, &pw, &ph);
    sliceFit(cell, r, &fx, &fy);

    // crosshair: the two other planes, projected into this view's slice uv, then into viewport uv
    float cu, cv;
    int lineU, lineV;  // plane drawn as the vertical / horizontal line
    if (cell == kAxial) { cu = cursor_.x; cv = 1.0f - cursor_.y; lineU = kSagittal; lineV = kCoronal; }
    else if (cell == kCoronal) { cu = cursor_.x; cv = cursor_.z; lineU = kSagittal; lineV = kAxial; }
    else { cu = cursor_.y; cv = cursor_.z; lineU = kCoronal; lineV = kAxial; }

    glViewport(r.x, height_ - r.y - r.h, r.w, r.h);
    sliceProg_.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, volTex_.get());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, maskTex_.get());
    sliceProg_.set("uVol", 0);
    sliceProg_.set("uMask", 1);
    sliceProg_.set("uOrigin", o.x, o.y, o.z);
    sliceProg_.set("uAxisU", u.x, u.y, u.z);
    sliceProg_.set("uAxisV", v.x, v.y, v.z);
    sliceProg_.set("uFit", fx, fy);
    sliceProg_.set("uCursor", (cu - 0.5f) / fx + 0.5f, (cv - 0.5f) / fy + 0.5f);
    sliceProg_.set("uPixel", 1.0f / r.w, 1.0f / r.h);
    sliceProg_.set("uLineU", kPlaneColor[lineU].x, kPlaneColor[lineU].y, kPlaneColor[lineU].z);
    sliceProg_.set("uLineV", kPlaneColor[lineV].x, kPlaneColor[lineV].y, kPlaneColor[lineV].z);
    sliceProg_.set("uBorder", kPlaneColor[cell].x, kPlaneColor[cell].y, kPlaneColor[cell].z);
    sliceProg_.set("uWindow", settings_.windowLo, std::max(settings_.windowHi, settings_.windowLo + 0.01f));
    sliceProg_.set("uTumourOn", settings_.tumourOn[0] ? 1.f : 0.f, settings_.tumourOn[1] ? 1.f : 0.f,
                   settings_.tumourOn[2] ? 1.f : 0.f);
    glUniform3fv(sliceProg_.loc("uTumourColor[0]"), 3, kTumourColor);
    sliceProg_.set("uTumourOpacity", settings_.tumourOpacity);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void Viewer::drawVolume(const Rect& r, float scale) {
    volTarget_.ensure(int(r.w * scale + 0.5f), int(r.h * scale + 0.5f));
    volTarget_.bind();
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    const vec3 eye = rot_.rotate({0, distance_, 0}), up = rot_.rotate({0, 0, 1});
    const mat4 proj = mat4::perspective(kFovY, float(r.w) / float(std::max(r.h, 1)), 0.05f, 20.0f);
    const mat4 invViewProj = (proj * mat4::lookAt(eye, {0, 0, 0}, up)).inverse();
    const int maxDim = std::max(vol_.dims[0], std::max(vol_.dims[1], vol_.dims[2]));
    const float voxel = 1.0f / float(maxDim);  // largest box side is 1 world unit
    const vec3 clipN = normalize(eye);

    volumeProg_.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, volTex_.get());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, maskTex_.get());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tfTex_.get());
    volumeProg_.set("uVol", 0);
    volumeProg_.set("uMask", 1);
    volumeProg_.set("uTF", 2);
    volumeProg_.setMat4("uInvViewProj", invViewProj.data());
    volumeProg_.set("uEye", eye.x, eye.y, eye.z);
    volumeProg_.set("uBoxHalf", boxHalf_.x, boxHalf_.y, boxHalf_.z);
    volumeProg_.set("uMode", settings_.mode);
    // coarser steps while interacting: fewer samples on top of fewer pixels
    volumeProg_.set("uStep", voxel * (scale < 1.0f ? 1.2f : 0.6f));
    volumeProg_.set("uRefStep", voxel);
    volumeProg_.set("uBrainOpacity", settings_.brainOpacity);
    volumeProg_.set("uTumourOpacity", settings_.tumourOpacity);
    volumeProg_.set("uTumourOn", settings_.tumourOn[0] ? 1.f : 0.f, settings_.tumourOn[1] ? 1.f : 0.f,
                    settings_.tumourOn[2] ? 1.f : 0.f);
    glUniform3fv(volumeProg_.loc("uTumourColor[0]"), 3, kTumourColor);
    volumeProg_.set("uTumourAlpha", kTumourAlpha.x, kTumourAlpha.y, kTumourAlpha.z);
    volumeProg_.set("uClipOn", settings_.clip < 0.999f ? 1 : 0);
    volumeProg_.set("uClipN", clipN.x, clipN.y, clipN.z);
    volumeProg_.set("uClipD", settings_.clip * 0.75f);  // 0.75 ~ half diagonal of the box
    volumeProg_.set("uWindow", settings_.windowLo, std::max(settings_.windowHi, settings_.windowLo + 0.01f));
    volumeProg_.set("uTexel", 1.0f / vol_.dims[0], 1.0f / vol_.dims[1], 1.0f / vol_.dims[2]);
    volumeProg_.set("uCursor", cursor_.x, cursor_.y, cursor_.z);
    volumeProg_.set("uShowPlanes", settings_.showPlanes && settings_.layout == 0 ? 1 : 0);
    volumeProg_.set("uSeed", seed_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void Viewer::blitVolume(const Rect& r) {
    glViewport(r.x, height_ - r.y - r.h, r.w, r.h);
    blitProg_.use();
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, volTarget_.texture());
    blitProg_.set("uTex", 2);
    blitProg_.set("uPixel", 1.0f / r.w, 1.0f / r.h);
    blitProg_.set("uBorder", kVolumeBorder.x, kVolumeBorder.y, kVolumeBorder.z);
    blitProg_.set("uDrawBorder", settings_.layout == 0 ? 1 : 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

std::string Viewer::stateJson() const {
    if (!hasVolume()) return "{\"loaded\":false}";
    const auto counts = vol_.labelCounts();
    const double ml = vol_.voxelMl();
    const int ix = std::min(int(cursor_.x * vol_.dims[0]), vol_.dims[0] - 1);
    const int iy = std::min(int(cursor_.y * vol_.dims[1]), vol_.dims[1] - 1);
    const int iz = std::min(int(cursor_.z * vol_.dims[2]), vol_.dims[2] - 1);
    const size_t idx = (size_t(iz) * vol_.dims[1] + iy) * vol_.dims[0] + ix;
    const int value = vol_.channels[size_t(settings_.channel)][idx];
    const int label = vol_.labels.empty() ? 0 : vol_.labels[idx];

    std::string channels;
    for (size_t i = 0; i < vol_.channelNames.size(); ++i)
        channels += (i ? ",\"" : "\"") + vol_.channelNames[i] + "\"";
    char buf[768];
    std::snprintf(buf, sizeof buf,
                  "{\"loaded\":true,\"dims\":[%d,%d,%d],\"spacing\":[%.3g,%.3g,%.3g],\"channels\":[%s],"
                  "\"tumour_ml\":[%.1f,%.1f,%.1f],\"cursor\":[%d,%d,%d],\"value\":%d,\"label\":%d,"
                  "\"render_scale\":%.2f}",
                  vol_.dims[0], vol_.dims[1], vol_.dims[2], vol_.spacing[0], vol_.spacing[1], vol_.spacing[2],
                  channels.c_str(), counts[1] * ml, counts[2] * ml, counts[3] * ml, ix, iy, iz, value, label,
                  lowResShown_ ? kLowResScale : 1.0f);
    return buf;
}

}  // namespace mv
