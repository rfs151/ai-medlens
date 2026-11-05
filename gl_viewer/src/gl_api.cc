#include "gl_api.h"

#if !defined(__EMSCRIPTEN__)
#define MV_GL_DEFINE(type, name) type mvgl_##name = nullptr;
MV_GL_FUNCTIONS(MV_GL_DEFINE)
#undef MV_GL_DEFINE

namespace mv {
bool loadGL(GLProcLoader loader, std::string* missing) {
    bool ok = true;
#define MV_GL_LOAD(type, name)                                   \
    mvgl_##name = reinterpret_cast<type>(loader(#name));         \
    if (!mvgl_##name) {                                          \
        ok = false;                                              \
        if (missing) *missing += std::string(#name) + " ";       \
    }
    MV_GL_FUNCTIONS(MV_GL_LOAD)
#undef MV_GL_LOAD
    return ok;
}
}  // namespace mv
#endif
