#pragma once
// GLSL sources shared by desktop (GLSL 3.30 core) and web (GLSL ES 3.00).
// The #version / precision header is prepended at compile time (MV_GLSL_HEADER), so these bodies
// must stay inside the common subset: no texture3D(), no implicit int->float, explicit precisions.

namespace mv::shaders {

// Full-viewport triangle generated from gl_VertexID: no vertex buffer at all.
inline constexpr const char* kFullscreenVS = R"GLSL(
out vec2 vUv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

// ---------------------------------------------------------------- GPU ray casting
// One ray per pixel: intersect the volume's bounding box, march front to back, look intensities up
// in a 1D transfer function, shade with the local gradient, and stop early once nearly opaque.
inline constexpr const char* kVolumeFS = R"GLSL(
in vec2 vUv;
out vec4 fragColor;

uniform mat4 uInvViewProj;
uniform vec3 uEye;
uniform vec3 uBoxHalf;        // half size of the volume box in world units (largest = 0.5)
uniform sampler3D uVol;       // R8 intensity, trilinear
uniform sampler3D uMask;      // RGBA8 soft tumour masks (R/G/B = class 1/2/3), trilinear
uniform sampler2D uTF;        // 256x1 RGBA transfer function
uniform int uMode;            // 0 = direct volume rendering, 1 = maximum intensity projection
uniform float uStep;          // ray step in world units
uniform float uRefStep;       // step the transfer-function opacity is defined for (one voxel)
uniform float uBrainOpacity;
uniform float uTumourOpacity;
uniform vec3 uTumourOn;       // 1/0 per tumour class
uniform vec3 uTumourColor[3];
uniform vec3 uTumourAlpha;     // per-class opacity weight (edema lower so the core shows through)
uniform vec3 uClipN;          // clip plane normal (points to the camera)
uniform float uClipD;         // keep points with dot(p, uClipN) <= uClipD
uniform int uClipOn;
uniform vec2 uWindow;         // grey window for MIP and the clip cap
uniform vec3 uTexel;          // 1 / volume dimensions
uniform vec3 uCursor;         // MPR cursor in texture space (drawn as coloured rings)
uniform int uShowPlanes;
uniform float uSeed;

const int MAX_STEPS = 1024;

// world (x right, y anterior, z superior) -> texture (x = i, y = j grows posterior, z = k)
vec3 toTex(vec3 p) {
    vec3 t = p / (2.0 * uBoxHalf) + 0.5;
    t.y = 1.0 - t.y;
    return t;
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233)) + uSeed) * 43758.5453); }

vec3 tumourMask(vec3 tc) { return texture(uMask, tc).rgb * uTumourOn; }

// Central-difference gradient of a scalar field, returned in world orientation.
vec3 gradVol(vec3 tc) {
    vec3 g = vec3(texture(uVol, tc + vec3(uTexel.x, 0, 0)).r - texture(uVol, tc - vec3(uTexel.x, 0, 0)).r,
                  texture(uVol, tc + vec3(0, uTexel.y, 0)).r - texture(uVol, tc - vec3(0, uTexel.y, 0)).r,
                  texture(uVol, tc + vec3(0, 0, uTexel.z)).r - texture(uVol, tc - vec3(0, 0, uTexel.z)).r);
    return vec3(g.x, -g.y, g.z) / uBoxHalf;
}
vec3 gradMask(vec3 tc) {
    vec3 on = uTumourOn;
    vec3 g = vec3(dot(texture(uMask, tc + vec3(uTexel.x, 0, 0)).rgb - texture(uMask, tc - vec3(uTexel.x, 0, 0)).rgb, on),
                  dot(texture(uMask, tc + vec3(0, uTexel.y, 0)).rgb - texture(uMask, tc - vec3(0, uTexel.y, 0)).rgb, on),
                  dot(texture(uMask, tc + vec3(0, 0, uTexel.z)).rgb - texture(uMask, tc - vec3(0, 0, uTexel.z)).rgb, on));
    return vec3(g.x, -g.y, g.z) / uBoxHalf;
}

vec3 tumourColor(vec3 m) {
    float s = m.r + m.g + m.b + 1e-4;
    return (m.r * uTumourColor[0] + m.g * uTumourColor[1] + m.b * uTumourColor[2]) / s;
}

// Opacity of the tumour at a sample: each class is a soft iso-surface of its smoothed mask.
// Edema (R) is drawn as a hollow shell around its boundary so the cores inside stay visible.
float tumourAlpha(vec3 m) {
    vec3 inside = smoothstep(vec3(0.35), vec3(0.65), m);
    vec3 a = inside * uTumourAlpha;
    a.r *= 1.0 - smoothstep(0.7, 0.95, m.r);
    return max(a.r, max(a.g, a.b)) * uTumourOpacity;
}

// Headlight Blinn-Phong; two-sided because gradients of MRI data have no consistent orientation.
vec3 shade(vec3 col, vec3 g, vec3 viewDir) {
    float len = length(g);
    if (len < 1e-3) return col * 0.8;
    vec3 n = g / len;
    vec3 l = -viewDir;
    if (dot(n, l) < 0.0) n = -n;
    float diff = dot(n, l);
    float spec = pow(max(dot(n, l), 0.0), 40.0);
    return col * (0.28 + 0.72 * diff) + vec3(0.22 * spec);
}

vec2 hitBox(vec3 o, vec3 d) {
    vec3 inv = 1.0 / (d + vec3(equal(d, vec3(0.0))) * 1e-7);
    vec3 t0 = (-uBoxHalf - o) * inv, t1 = (uBoxHalf - o) * inv;
    vec3 tmin = min(t0, t1), tmax = max(t0, t1);
    return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

void main() {
    vec4 farP = uInvViewProj * vec4(vUv * 2.0 - 1.0, 1.0, 1.0);
    vec3 o = uEye;
    vec3 d = normalize(farP.xyz / farP.w - uEye);
    vec2 span = hitBox(o, d);
    float tn = max(span.x, 0.0), tf = span.y;

    // analytic clip against the camera-facing plane: exact cut surface, no wasted samples
    bool capped = false;
    if (uClipOn == 1) {
        float dn = dot(d, uClipN);
        float tp = (uClipD - dot(o, uClipN)) / (abs(dn) > 1e-6 ? dn : 1e-6);
        if (dn < 0.0) {
            if (tp > tn) { tn = tp; capped = true; }
        } else {
            tf = min(tf, tp);
        }
    }
    if (tf <= tn) { fragColor = vec4(0.0); return; }

    // jittered start hides the "wood grain" banding of fixed steps
    float t = tn + uStep * hash(gl_FragCoord.xy);
    vec4 acc = vec4(0.0);
    float vmax = 0.0;
    vec3 mmax = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; ++i) {
        if (t > tf) break;
        vec3 tc = toTex(o + d * t);
        float v = texture(uVol, tc).r;
        vec3 m = tumourMask(tc);
        t += uStep;

        if (uMode == 1) { vmax = max(vmax, v); mmax = max(mmax, m); continue; }

        float mt = max(m.r, max(m.g, m.b));
        float ta = tumourAlpha(m);

        if (capped) {  // first sample on the cut face: show it as an opaque oblique slice
            capped = false;
            if (v > 0.004) {
                vec3 c = vec3(clamp((v - uWindow.x) / (uWindow.y - uWindow.x), 0.0, 1.0));
                if (ta > 0.0) c = mix(c, tumourColor(m), 0.65 * smoothstep(0.35, 0.65, mt));
                acc = vec4(c, 1.0);
                break;
            }
        }

        vec4 tfc = texture(uTF, vec2(v, 0.5));
        float ba = tfc.a * uBrainOpacity;
        vec3 g = vec3(0.0);
        if (ta > 0.0) {
            g = gradMask(tc);
        } else if (ba > 0.002) {
            // boundary emphasis: tissue interfaces (high gradient) opaque, homogeneous tissue faint
            g = gradVol(tc);
            ba *= 0.2 + 1.3 * smoothstep(0.04, 0.35, length(g));
        }
        vec3 col = tfc.rgb;
        float a = ba;
        if (ta > 0.0) {
            col = mix(col, tumourColor(m * uTumourAlpha), ta / max(ta + ba, 1e-4));
            a = max(ba, ta);
        }
        if (uShowPlanes == 1 && ta == 0.0 && length(g) > 0.25) {
            // MPR plane positions as contour lines where the planes cross tissue boundaries
            vec3 dist = abs(tc - uCursor) / uTexel;
            if (dist.z < 0.7) { col = vec3(0.95, 0.35, 0.35); a = max(a, 0.5); }
            else if (dist.y < 0.7) { col = vec3(0.35, 0.9, 0.45); a = max(a, 0.5); }
            else if (dist.x < 0.7) { col = vec3(0.98, 0.82, 0.3); a = max(a, 0.5); }
        }
        if (a < 0.002) continue;

        col = shade(col, g, d);
        a = 1.0 - pow(1.0 - min(a, 0.995), uStep / uRefStep);  // opacity correction for the step size
        acc.rgb += (1.0 - acc.a) * a * col;
        acc.a += (1.0 - acc.a) * a;
        if (acc.a > 0.97) break;  // early ray termination
    }

    if (uMode == 1) {
        vec3 c = vec3(clamp((vmax - uWindow.x) / (uWindow.y - uWindow.x), 0.0, 1.0));
        float mt = max(mmax.r, max(mmax.g, mmax.b));
        c = mix(c, tumourColor(mmax), 0.6 * smoothstep(0.35, 0.65, mt) * step(0.001, uTumourOpacity));
        acc = vec4(c, vmax > 0.004 ? 1.0 : 0.0);
    }
    fragColor = acc;  // premultiplied alpha
}
)GLSL";

// ---------------------------------------------------------------- multi-planar reconstruction
inline constexpr const char* kSliceFS = R"GLSL(
in vec2 vUv;
out vec4 fragColor;

uniform sampler3D uVol;
uniform sampler3D uMask;
uniform vec3 uOrigin;         // texture coordinate at slice uv (0,0)
uniform vec3 uAxisU;          // texture delta per unit slice u
uniform vec3 uAxisV;          // texture delta per unit slice v
uniform vec2 uFit;            // viewport uv -> slice uv scale (keeps physical aspect)
uniform vec2 uCursor;         // crosshair position in viewport uv
uniform vec2 uPixel;          // size of one pixel in viewport uv
uniform vec3 uLineU;          // colour of the vertical crosshair line
uniform vec3 uLineV;          // colour of the horizontal crosshair line
uniform vec3 uBorder;
uniform vec2 uWindow;
uniform vec3 uTumourOn;
uniform vec3 uTumourColor[3];
uniform float uTumourOpacity;

void main() {
    vec2 s = (vUv - 0.5) * uFit + 0.5;
    vec3 col = vec3(0.0);
    if (all(greaterThanEqual(s, vec2(0.0))) && all(lessThanEqual(s, vec2(1.0)))) {
        vec3 tc = uOrigin + s.x * uAxisU + s.y * uAxisV;
        float v = texture(uVol, tc).r;
        col = vec3(clamp((v - uWindow.x) / (uWindow.y - uWindow.x), 0.0, 1.0));
        vec3 m = texture(uMask, tc).rgb * uTumourOn;
        float mt = max(m.r, max(m.g, m.b));
        if (mt > 0.01 && uTumourOpacity > 0.0) {
            float sm = m.r + m.g + m.b + 1e-4;
            vec3 tcol = (m.r * uTumourColor[0] + m.g * uTumourColor[1] + m.b * uTumourColor[2]) / sm;
            float fill = smoothstep(0.35, 0.65, mt) * 0.35;
            float edge = 1.0 - smoothstep(0.0, 1.5 * fwidth(mt) + 1e-4, abs(mt - 0.5));  // iso-line at 0.5
            col = mix(col, tcol, clamp(fill + edge * 0.9, 0.0, 1.0) * uTumourOpacity);
        }
    }
    vec2 dpx = abs(vUv - uCursor) / uPixel;
    if (dpx.x < 0.8 && dpx.y > 6.0) col = mix(col, uLineU, 0.85);
    if (dpx.y < 0.8 && dpx.x > 6.0) col = mix(col, uLineV, 0.85);
    vec2 e = min(vUv, 1.0 - vUv) / uPixel;
    if (min(e.x, e.y) < 2.0) col = uBorder;
    fragColor = vec4(col, 1.0);
}
)GLSL";

// ---------------------------------------------------------------- composite volume target into the layout
inline constexpr const char* kBlitFS = R"GLSL(
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uTex;
uniform vec2 uPixel;
uniform vec3 uBorder;
uniform int uDrawBorder;
void main() {
    vec4 c = texture(uTex, vUv);  // premultiplied
    vec3 bg = mix(vec3(0.03, 0.04, 0.06), vec3(0.11, 0.14, 0.19), vUv.y);
    vec3 col = c.rgb + (1.0 - c.a) * bg;
    vec2 e = min(vUv, 1.0 - vUv) / uPixel;
    if (uDrawBorder == 1 && min(e.x, e.y) < 2.0) col = uBorder;
    fragColor = vec4(col, 1.0);
}
)GLSL";

}  // namespace mv::shaders
