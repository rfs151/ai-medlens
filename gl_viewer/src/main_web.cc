// Web entry point (Emscripten -> WebAssembly + WebGL2). Same Viewer as the desktop build; this file
// only owns the WebGL context, DOM input events and the requestAnimationFrame loop, and exposes a
// small C API that the page's HTML controls call:
//
//   mv_load_volume(ptr, len)        uncompressed .mlvol bytes in the wasm heap -> 0 on success
//   mv_set_int(key, value)          mode | channel | preset | layout | planes | tumour0..2
//   mv_set_float(key, value)        brain | tumour | clip | winlo | winhi
//   mv_reset_view()
//   mv_state_json()                 volume facts + cursor readout as JSON

#include <emscripten/emscripten.h>
#include <emscripten/html5.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "gl_api.h"
#include "viewer.h"

namespace {

constexpr const char* kCanvas = "#mvCanvas";
mv::Viewer* g_viewer = nullptr;
double g_dpr = 1.0;
float g_pinchDist = 0;

double nowSec() { return emscripten_get_now() / 1000.0; }

EM_JS(void, mv_canvas_origin, (double* out), {
    const r = document.querySelector('#mvCanvas').getBoundingClientRect();
    HEAPF64[out >> 3] = r.left;
    HEAPF64[(out >> 3) + 1] = r.top;
});

EM_JS(void, mv_report_error, (const char* msg), {
    if (Module.onViewerError) Module.onViewerError(UTF8ToString(msg));
});

// Client (CSS px, page) -> framebuffer pixels of the canvas.
void toCanvas(double clientX, double clientY, float* x, float* y) {
    double o[2];
    mv_canvas_origin(o);
    *x = float((clientX - o[0]) * g_dpr);
    *y = float((clientY - o[1]) * g_dpr);
}

// Keep the drawing buffer at CSS size x devicePixelRatio so the page stays sharp on HiDPI screens.
void syncCanvasSize() {
    double cssW = 0, cssH = 0;
    emscripten_get_element_css_size(kCanvas, &cssW, &cssH);
    g_dpr = emscripten_get_device_pixel_ratio();
    const int w = std::max(1, int(cssW * g_dpr + 0.5)), h = std::max(1, int(cssH * g_dpr + 0.5));
    int cw = 0, ch = 0;
    emscripten_get_canvas_element_size(kCanvas, &cw, &ch);
    if (cw != w || ch != h) emscripten_set_canvas_element_size(kCanvas, w, h);
    g_viewer->resize(w, h);
}

void frame() {
    syncCanvasSize();
    const double t = nowSec();
    if (g_viewer->needsRedraw(t)) g_viewer->render(t, 0);  // idle frames issue no GL calls at all
}

EM_BOOL onMouseDown(int, const EmscriptenMouseEvent* e, void*) {
    float x, y;
    toCanvas(e->clientX, e->clientY, &x, &y);
    g_viewer->pointerDown(x, y, e->button, nowSec());
    return EM_TRUE;
}
EM_BOOL onMouseMove(int, const EmscriptenMouseEvent* e, void*) {
    float x, y;
    toCanvas(e->clientX, e->clientY, &x, &y);
    g_viewer->pointerMove(x, y, nowSec());
    return EM_FALSE;
}
EM_BOOL onMouseUp(int, const EmscriptenMouseEvent*, void*) {
    g_viewer->pointerUp(nowSec());
    return EM_FALSE;
}
EM_BOOL onWheel(int, const EmscriptenWheelEvent* e, void*) {
    float x, y;
    toCanvas(e->mouse.clientX, e->mouse.clientY, &x, &y);
    if (e->deltaY != 0) g_viewer->wheel(x, y, e->deltaY < 0 ? 1.0f : -1.0f, nowSec());
    return EM_TRUE;  // keep the page from scrolling while over the viewer
}

// One finger drags (rotate / pick), two fingers pinch-zoom the 3D view.
EM_BOOL onTouch(int type, const EmscriptenTouchEvent* e, void*) {
    const double t = nowSec();
    if (e->numTouches >= 2) {
        float x0, y0, x1, y1;
        toCanvas(e->touches[0].clientX, e->touches[0].clientY, &x0, &y0);
        toCanvas(e->touches[1].clientX, e->touches[1].clientY, &x1, &y1);
        const float dist = std::hypot(x1 - x0, y1 - y0);
        if (type == EMSCRIPTEN_EVENT_TOUCHMOVE && g_pinchDist > 0 && std::fabs(dist - g_pinchDist) > 12) {
            g_viewer->pointerUp(t);
            g_viewer->wheel((x0 + x1) / 2, (y0 + y1) / 2, dist > g_pinchDist ? 1.0f : -1.0f, t);
            g_pinchDist = dist;
        } else if (type == EMSCRIPTEN_EVENT_TOUCHSTART) {
            g_pinchDist = dist;
        }
        return EM_TRUE;
    }
    g_pinchDist = 0;
    if (e->numTouches < 1) return EM_TRUE;
    float x, y;
    toCanvas(e->touches[0].clientX, e->touches[0].clientY, &x, &y);
    if (type == EMSCRIPTEN_EVENT_TOUCHSTART) g_viewer->pointerDown(x, y, 0, t);
    else if (type == EMSCRIPTEN_EVENT_TOUCHMOVE) g_viewer->pointerMove(x, y, t);
    else g_viewer->pointerUp(t);
    return EM_TRUE;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int mv_load_volume(const uint8_t* data, int len) {
    mv::Volume v;
    std::string err;
    if (!g_viewer || !v.parse(data, size_t(len), &err) || !g_viewer->setVolume(std::move(v), &err)) {
        mv_report_error(("volume: " + err).c_str());
        return 1;
    }
    g_viewer->settingsChanged(-100);
    return 0;
}

EMSCRIPTEN_KEEPALIVE void mv_set_int(const char* key, int value) {
    if (!g_viewer) return;
    auto& s = g_viewer->settings();
    if (!std::strcmp(key, "mode")) s.mode = value;
    else if (!std::strcmp(key, "channel")) s.channel = value;
    else if (!std::strcmp(key, "preset")) s.preset = value;
    else if (!std::strcmp(key, "layout")) s.layout = value;
    else if (!std::strcmp(key, "planes")) s.showPlanes = value != 0;
    else if (!std::strncmp(key, "tumour", 6) && key[6] >= '0' && key[6] <= '2') s.tumourOn[key[6] - '0'] = value != 0;
    else return;
    g_viewer->settingsChanged(nowSec());
}

EMSCRIPTEN_KEEPALIVE void mv_set_float(const char* key, float value) {
    if (!g_viewer) return;
    auto& s = g_viewer->settings();
    if (!std::strcmp(key, "brain")) s.brainOpacity = value;
    else if (!std::strcmp(key, "tumour")) s.tumourOpacity = value;
    else if (!std::strcmp(key, "clip")) s.clip = value;
    else if (!std::strcmp(key, "winlo")) s.windowLo = value;
    else if (!std::strcmp(key, "winhi")) s.windowHi = value;
    else return;
    g_viewer->settingsChanged(nowSec());
}

EMSCRIPTEN_KEEPALIVE void mv_reset_view() {
    if (g_viewer) {
        g_viewer->resetView();
        g_viewer->settingsChanged(nowSec());
    }
}

EMSCRIPTEN_KEEPALIVE const char* mv_state_json() {
    static std::string json;
    json = g_viewer ? g_viewer->stateJson() : "{\"loaded\":false}";
    return json.c_str();
}

}  // extern "C"

int main() {
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.majorVersion = 2;  // WebGL2 == OpenGL ES 3.0: 3D textures, GLSL ES 3.00
    attrs.minorVersion = 0;
    attrs.alpha = EM_FALSE;
    attrs.depth = EM_FALSE;
    attrs.stencil = EM_FALSE;
    attrs.antialias = EM_FALSE;
    attrs.powerPreference = EM_WEBGL_POWER_PREFERENCE_HIGH_PERFORMANCE;
    const EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx = emscripten_webgl_create_context(kCanvas, &attrs);
    if (ctx <= 0) {
        mv_report_error("WebGL2 is not available in this browser");
        return 0;
    }
    emscripten_webgl_make_context_current(ctx);

    static mv::Viewer viewer;
    std::string log;
    if (!viewer.init(&log)) {
        mv_report_error(("shader: " + log).c_str());
        return 0;
    }
    g_viewer = &viewer;

    emscripten_set_mousedown_callback(kCanvas, nullptr, EM_TRUE, onMouseDown);
    // move/up on the document so a drag keeps working when the pointer leaves the canvas
    emscripten_set_mousemove_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, EM_FALSE, onMouseMove);
    emscripten_set_mouseup_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, EM_FALSE, onMouseUp);
    emscripten_set_wheel_callback(kCanvas, nullptr, EM_FALSE, onWheel);
    emscripten_set_touchstart_callback(kCanvas, nullptr, EM_FALSE, onTouch);
    emscripten_set_touchmove_callback(kCanvas, nullptr, EM_FALSE, onTouch);
    emscripten_set_touchend_callback(kCanvas, nullptr, EM_FALSE, onTouch);
    emscripten_set_touchcancel_callback(kCanvas, nullptr, EM_FALSE, onTouch);

    EM_ASM({ if (Module.onViewerReady) Module.onViewerReady(); });
    emscripten_set_main_loop(frame, 0, EM_FALSE);  // requestAnimationFrame-driven
    return 0;
}
