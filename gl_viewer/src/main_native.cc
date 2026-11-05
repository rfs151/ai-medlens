// Desktop entry point: GLFW window + OpenGL 3.3 core (Linux / Windows / macOS).
//
//   medlens_volume <volume.mlvol[.gz]>                      interactive window
//   medlens_volume <volume> --screenshot out.png [--size 1280x800]   render once off-screen, save PNG
//   medlens_volume <volume> --bench 50                      time full-resolution frames (glFinish)
//   options: --layout 0|1  --mode 0|1  --preset 0..2  --clip -1..1  --channel n  --opacity 0..1
//
// Keys: 1/2 volume rendering / MIP, C channel, P preset, T tumour on/off, L layout, R reset view,
//       Up/Down brain opacity, [ ] cut plane, S save screenshot.

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <zlib.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gl_api.h"
#include "viewer.h"

namespace {

bool readFile(const char* path, std::vector<uint8_t>* out) {
    gzFile f = gzopen(path, "rb");  // reads plain and gzip-compressed files alike
    if (!f) return false;
    uint8_t buf[1 << 16];
    int n;
    while ((n = gzread(f, buf, sizeof buf)) > 0) out->insert(out->end(), buf, buf + n);
    gzclose(f);
    return n == 0;
}

void putU32(std::vector<uint8_t>& v, uint32_t x) {
    for (int s = 24; s >= 0; s -= 8) v.push_back(uint8_t(x >> s));
}

void pngChunk(std::vector<uint8_t>& png, const char* type, const std::vector<uint8_t>& data) {
    putU32(png, uint32_t(data.size()));
    const size_t start = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), data.begin(), data.end());
    putU32(png, uint32_t(crc32(0, png.data() + start, uInt(png.size() - start))));
}

// Reads the bound framebuffer (bottom-up RGBA) and writes an RGB PNG using zlib.
bool savePng(const char* path, int w, int h) {
    std::vector<uint8_t> rgba(size_t(w) * h * 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    std::vector<uint8_t> raw;
    raw.reserve(size_t(h) * (w * 3 + 1));
    for (int y = h - 1; y >= 0; --y) {  // PNG rows go top-down
        raw.push_back(0);                // filter: none
        for (int x = 0; x < w; ++x) raw.insert(raw.end(), &rgba[(size_t(y) * w + x) * 4], &rgba[(size_t(y) * w + x) * 4 + 3]);
    }
    uLongf zlen = compressBound(uLong(raw.size()));
    std::vector<uint8_t> z(zlen);
    if (compress2(z.data(), &zlen, raw.data(), uLong(raw.size()), 6) != Z_OK) return false;
    z.resize(zlen);

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<uint8_t> ihdr;
    putU32(ihdr, uint32_t(w));
    putU32(ihdr, uint32_t(h));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8-bit, RGB, deflate, no filter, no interlace
    pngChunk(png, "IHDR", ihdr);
    pngChunk(png, "IDAT", z);
    pngChunk(png, "IEND", {});
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const bool ok = std::fwrite(png.data(), 1, png.size(), f) == png.size();
    std::fclose(f);
    return ok;
}

struct App {
    GLFWwindow* window = nullptr;
    std::unique_ptr<mv::Viewer> viewerPtr = std::make_unique<mv::Viewer>();
    mv::Viewer& viewer = *viewerPtr;
    double scaleX = 1, scaleY = 1;  // window coords -> framebuffer pixels (HiDPI)
};

double now() { return glfwGetTime(); }

void updateTitle(App& app) {
    const auto& s = app.viewer.settings();
    char title[256];
    std::snprintf(title, sizeof title, "MedLens Volume  |  %s  preset %d  opacity %.2f  cut %.2f  |  1/2 C P T L R [ ] S",
                  s.mode ? "MIP" : "DVR", s.preset, s.brainOpacity, s.clip);
    glfwSetWindowTitle(app.window, title);
}

void onKey(GLFWwindow* w, int key, int, int action, int) {
    if (action == GLFW_RELEASE) return;
    auto& app = *static_cast<App*>(glfwGetWindowUserPointer(w));
    auto& s = app.viewer.settings();
    switch (key) {
        case GLFW_KEY_1: s.mode = 0; break;
        case GLFW_KEY_2: s.mode = 1; break;
        case GLFW_KEY_C: s.channel = (s.channel + 1) % 3; break;
        case GLFW_KEY_P: s.preset = (s.preset + 1) % 3; break;
        case GLFW_KEY_T: s.tumourOpacity = s.tumourOpacity > 0 ? 0.0f : 0.9f; break;
        case GLFW_KEY_L: s.layout = 1 - s.layout; break;
        case GLFW_KEY_R: app.viewer.resetView(); break;
        case GLFW_KEY_UP: s.brainOpacity = std::min(1.0f, s.brainOpacity + 0.05f); break;
        case GLFW_KEY_DOWN: s.brainOpacity = std::max(0.0f, s.brainOpacity - 0.05f); break;
        case GLFW_KEY_LEFT_BRACKET: s.clip = std::max(-1.0f, s.clip - 0.05f); break;
        case GLFW_KEY_RIGHT_BRACKET: s.clip = std::min(1.0f, s.clip + 0.05f); break;
        case GLFW_KEY_S: {
            int fw, fh;
            glfwGetFramebufferSize(w, &fw, &fh);
            app.viewer.render(now() + 10, 0);  // +10 s: force the full-resolution path
            std::printf(savePng("medlens_screenshot.png", fw, fh) ? "saved medlens_screenshot.png\n" : "save failed\n");
            return;
        }
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, 1); return;
        default: return;
    }
    app.viewer.settingsChanged(now());
    updateTitle(app);
}

}  // namespace

int main(int argc, char** argv) {
    const char* volumePath = nullptr;
    const char* screenshot = nullptr;
    int width = 1280, height = 800, bench = 0;
    mv::ViewSettings settings;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (!std::strcmp(argv[i], "--screenshot")) screenshot = next();
        else if (!std::strcmp(argv[i], "--size")) std::sscanf(next(), "%dx%d", &width, &height);
        else if (!std::strcmp(argv[i], "--bench")) bench = std::atoi(next());
        else if (!std::strcmp(argv[i], "--layout")) settings.layout = std::atoi(next());
        else if (!std::strcmp(argv[i], "--mode")) settings.mode = std::atoi(next());
        else if (!std::strcmp(argv[i], "--preset")) settings.preset = std::atoi(next());
        else if (!std::strcmp(argv[i], "--clip")) settings.clip = float(std::atof(next()));
        else if (!std::strcmp(argv[i], "--channel")) settings.channel = std::atoi(next());
        else if (!std::strcmp(argv[i], "--opacity")) settings.brainOpacity = float(std::atof(next()));
        else volumePath = argv[i];
    }
    if (!volumePath) {
        std::fprintf(stderr, "usage: %s volume.mlvol[.gz] [--screenshot out.png] [--bench N] [--size WxH]\n", argv[0]);
        return 1;
    }
    std::vector<uint8_t> bytes;
    mv::Volume volume;
    std::string err;
    if (!readFile(volumePath, &bytes) || !volume.parse(bytes.data(), bytes.size(), &err)) {
        std::fprintf(stderr, "cannot load %s %s\n", volumePath, err.c_str());
        return 1;
    }
    bytes = {};

    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);  // required on macOS
    const bool offscreen = screenshot || bench > 0;
    if (offscreen) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    App app;
    app.window = glfwCreateWindow(width, height, "MedLens Volume", nullptr, nullptr);
    if (!app.window) {
        std::fprintf(stderr, "cannot create an OpenGL 3.3 core window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(app.window);
    glfwSwapInterval(1);
    std::string missing;
    if (!mv::loadGL(reinterpret_cast<mv::GLProcLoader>(glfwGetProcAddress), &missing)) {
        std::fprintf(stderr, "missing GL functions: %s\n", missing.c_str());
        return 1;
    }
    std::printf("GL %s | %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

    if (!app.viewer.init(&err) || !app.viewer.setVolume(std::move(volume), &err)) {
        std::fprintf(stderr, "init failed: %s\n", err.c_str());
        return 1;
    }
    app.viewer.settings() = settings;
    app.viewer.settingsChanged(-100);  // far in the past: first frame renders at full resolution

    if (offscreen) {
        // Hidden windows have no reliable default framebuffer, so render into our own FBO.
        mv::RenderTarget target;
        if (!target.ensure(width, height)) return 1;
        app.viewer.resize(width, height);
        app.viewer.render(1e6, target.framebuffer());  // warm-up (shader compile, texture upload)
        if (bench > 0) {
            double total = 0, best = 1e9;
            for (int i = 0; i < bench; ++i) {
                app.viewer.resetView();  // marks the volume dirty -> full ray casting every frame
                glFinish();
                auto t0 = std::chrono::steady_clock::now();
                app.viewer.render(1e6, target.framebuffer());
                glFinish();
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                total += ms;
                best = std::min(best, ms);
            }
            std::printf("%dx%d full-resolution frame: avg %.1f ms, best %.1f ms\n", width, height, total / bench, best);
        }
        if (screenshot) {
            app.viewer.resetView();
            app.viewer.render(1e6, target.framebuffer());
            if (!savePng(screenshot, width, height)) return 1;
            std::printf("saved %s\n", screenshot);
        }
        target = {};
        app.viewerPtr.reset();  // free GL objects while the context still exists
        glfwTerminate();
        return 0;
    }

    glfwSetWindowUserPointer(app.window, &app);
    glfwSetKeyCallback(app.window, onKey);
    glfwSetMouseButtonCallback(app.window, [](GLFWwindow* w, int button, int action, int) {
        auto& a = *static_cast<App*>(glfwGetWindowUserPointer(w));
        double x, y;
        glfwGetCursorPos(w, &x, &y);
        if (action == GLFW_PRESS) a.viewer.pointerDown(float(x * a.scaleX), float(y * a.scaleY), button, now());
        else a.viewer.pointerUp(now());
    });
    glfwSetCursorPosCallback(app.window, [](GLFWwindow* w, double x, double y) {
        auto& a = *static_cast<App*>(glfwGetWindowUserPointer(w));
        a.viewer.pointerMove(float(x * a.scaleX), float(y * a.scaleY), now());
    });
    glfwSetScrollCallback(app.window, [](GLFWwindow* w, double, double dy) {
        auto& a = *static_cast<App*>(glfwGetWindowUserPointer(w));
        double x, y;
        glfwGetCursorPos(w, &x, &y);
        a.viewer.wheel(float(x * a.scaleX), float(y * a.scaleY), float(dy), now());
    });
    updateTitle(app);

    while (!glfwWindowShouldClose(app.window)) {
        int fw, fh, ww, wh;
        glfwGetFramebufferSize(app.window, &fw, &fh);
        glfwGetWindowSize(app.window, &ww, &wh);
        app.scaleX = ww > 0 ? double(fw) / ww : 1;
        app.scaleY = wh > 0 ? double(fh) / wh : 1;
        app.viewer.resize(fw, fh);
        if (app.viewer.needsRedraw(now())) {
            app.viewer.render(now(), 0);
            glfwSwapBuffers(app.window);
            glfwPollEvents();
        } else {
            glfwWaitEventsTimeout(0.05);  // idle: no busy loop, wake up for the full-res refinement
        }
    }
    app.viewerPtr.reset();
    glfwTerminate();
    return 0;
}
