#pragma once
// Platform-independent viewer: owns GL resources, layout, camera and interaction state.
// The platform layer (main_native.cc / main_web.cc) only creates the context, forwards input in
// framebuffer pixels (origin top-left) and calls render() when needsRedraw() says so.

#include <string>

#include "gl_objects.h"
#include "vecmath.h"
#include "volume.h"

namespace mv {

struct ViewSettings {
    int mode = 0;                 // 0 = volume rendering, 1 = maximum intensity projection
    int channel = 0;              // index into Volume::channels
    int preset = 1;               // transfer function preset, see kPresetNames
    float brainOpacity = 0.3f;    // scales the transfer function alpha
    float tumourOpacity = 0.9f;   // 0 hides the tumour overlay everywhere
    bool tumourOn[3] = {true, true, true};
    float clip = 1.0f;            // camera-facing cut plane: 1 = off, -1 = cut everything
    float windowLo = 0.08f;       // intensity window, 0..1 of the 8-bit range
    float windowHi = 0.92f;
    int layout = 0;               // 0 = 2x2 (axial / 3D / coronal / sagittal), 1 = 3D only
    bool showPlanes = true;       // draw MPR slice positions on the 3D view
};

class Viewer {
public:
    static constexpr const char* kPresetNames[3] = {"grey", "warm", "surface"};

    bool init(std::string* log);
    bool setVolume(Volume volume, std::string* error);
    bool hasVolume() const { return vol_.voxelCount() > 0; }

    void resize(int width, int height);
    // Draws the whole layout into framebuffer `target` (0 = default framebuffer).
    void render(double now, GLuint target = 0);
    // True while something changed or a reduced-resolution frame still needs its full-res refinement.
    bool needsRedraw(double now) const;

    void pointerDown(float x, float y, int button, double now);
    void pointerMove(float x, float y, double now);
    void pointerUp(double now);
    void wheel(float x, float y, float delta, double now);

    void resetView();
    ViewSettings& settings() { return settings_; }
    // Call after changing settings(); re-uploads only what the change requires.
    void settingsChanged(double now);

    // Volume facts and cursor readout for the UI, as a JSON object.
    std::string stateJson() const;

private:
    enum Cell { kAxial = 0, kCoronal = 1, kSagittal = 2, kVolume = 3 };
    struct Rect { int x = 0, y = 0, w = 0, h = 0; };  // pixels, origin top-left

    Rect cellRect(int cell) const;
    bool cellVisible(int cell) const { return settings_.layout == 0 || cell == kVolume; }
    int cellAt(float x, float y) const;

    void sliceBasis(int cell, vec3* origin, vec3* u, vec3* v, float* physW, float* physH) const;
    void sliceFit(int cell, const Rect& r, float* fx, float* fy) const;
    void pickSlice(int cell, float x, float y);

    void drawSlice(int cell, const Rect& r);
    void drawVolume(const Rect& r, float scale);
    void blitVolume(const Rect& r);
    void uploadChannel();
    void uploadTransferFunction();
    void touch(double now) { lastInteraction_ = now; }

    Program volumeProg_, sliceProg_, blitProg_;
    VertexArray vao_;
    Texture volTex_, maskTex_, tfTex_;
    RenderTarget volTarget_;

    Volume vol_;
    ViewSettings settings_;
    int uploadedChannel_ = -1;
    vec3 boxHalf_{0.5f, 0.5f, 0.5f};
    vec3 cursor_{0.5f, 0.5f, 0.5f};  // texture space

    quat rot_;
    float distance_ = 1.9f;
    int width_ = 1, height_ = 1;

    int dragCell_ = -1, dragButton_ = 0;
    float lastX_ = 0, lastY_ = 0;
    double lastInteraction_ = -1e9;
    bool volumeDirty_ = true, slicesDirty_ = true;
    bool lowResShown_ = false;
    float seed_ = 0;
};

}  // namespace mv
