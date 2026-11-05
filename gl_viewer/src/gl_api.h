#pragma once
// One OpenGL include for every platform.
//  - Web (Emscripten): WebGL2 == OpenGL ES 3.0, functions are linked directly.
//  - Desktop (Linux / Windows / macOS): OpenGL 3.3 core. Function pointers are loaded at runtime
//    by gl_api.cc through the window toolkit's getProcAddress, so no GLEW/glad dependency.
// Shaders are written once in the common subset of GLSL 3.30 and GLSL ES 3.00; only the header differs.

#include <string>

#if defined(__EMSCRIPTEN__)
#include <GLES3/gl3.h>
#define MV_GLSL_HEADER "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler3D;\n"
#else
#include <GL/glcorearb.h>  // Khronos header: types and enums only (prototypes stay off)

#define MV_GLSL_HEADER "#version 330 core\n"

// Every GL entry point the viewer uses. The pointers are named mvgl_* and #defined back to gl*,
// so they never clash with symbols exported by the system's libGL / opengl32.dll.
#define MV_GL_FUNCTIONS(X)                                   \
    X(PFNGLACTIVETEXTUREPROC, glActiveTexture)               \
    X(PFNGLATTACHSHADERPROC, glAttachShader)                 \
    X(PFNGLBINDFRAMEBUFFERPROC, glBindFramebuffer)           \
    X(PFNGLBINDTEXTUREPROC, glBindTexture)                   \
    X(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray)           \
    X(PFNGLBLENDFUNCPROC, glBlendFunc)                       \
    X(PFNGLCHECKFRAMEBUFFERSTATUSPROC, glCheckFramebufferStatus) \
    X(PFNGLCLEARPROC, glClear)                               \
    X(PFNGLCLEARCOLORPROC, glClearColor)                     \
    X(PFNGLCOMPILESHADERPROC, glCompileShader)               \
    X(PFNGLCREATEPROGRAMPROC, glCreateProgram)               \
    X(PFNGLCREATESHADERPROC, glCreateShader)                 \
    X(PFNGLDELETEFRAMEBUFFERSPROC, glDeleteFramebuffers)     \
    X(PFNGLDELETEPROGRAMPROC, glDeleteProgram)               \
    X(PFNGLDELETESHADERPROC, glDeleteShader)                 \
    X(PFNGLDELETETEXTURESPROC, glDeleteTextures)             \
    X(PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays)     \
    X(PFNGLDISABLEPROC, glDisable)                           \
    X(PFNGLDRAWARRAYSPROC, glDrawArrays)                     \
    X(PFNGLENABLEPROC, glEnable)                             \
    X(PFNGLFINISHPROC, glFinish)                             \
    X(PFNGLFRAMEBUFFERTEXTURE2DPROC, glFramebufferTexture2D) \
    X(PFNGLGENFRAMEBUFFERSPROC, glGenFramebuffers)           \
    X(PFNGLGENTEXTURESPROC, glGenTextures)                   \
    X(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays)           \
    X(PFNGLGETERRORPROC, glGetError)                         \
    X(PFNGLGETINTEGERVPROC, glGetIntegerv)                   \
    X(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog)       \
    X(PFNGLGETPROGRAMIVPROC, glGetProgramiv)                 \
    X(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog)         \
    X(PFNGLGETSHADERIVPROC, glGetShaderiv)                   \
    X(PFNGLGETSTRINGPROC, glGetString)                       \
    X(PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation)     \
    X(PFNGLLINKPROGRAMPROC, glLinkProgram)                   \
    X(PFNGLPIXELSTOREIPROC, glPixelStorei)                   \
    X(PFNGLREADPIXELSPROC, glReadPixels)                     \
    X(PFNGLSCISSORPROC, glScissor)                           \
    X(PFNGLSHADERSOURCEPROC, glShaderSource)                 \
    X(PFNGLTEXIMAGE2DPROC, glTexImage2D)                     \
    X(PFNGLTEXIMAGE3DPROC, glTexImage3D)                     \
    X(PFNGLTEXPARAMETERIPROC, glTexParameteri)               \
    X(PFNGLTEXSUBIMAGE2DPROC, glTexSubImage2D)               \
    X(PFNGLUNIFORM1FPROC, glUniform1f)                       \
    X(PFNGLUNIFORM1IPROC, glUniform1i)                       \
    X(PFNGLUNIFORM2FPROC, glUniform2f)                       \
    X(PFNGLUNIFORM3FPROC, glUniform3f)                       \
    X(PFNGLUNIFORM3FVPROC, glUniform3fv)                     \
    X(PFNGLUNIFORMMATRIX4FVPROC, glUniformMatrix4fv)         \
    X(PFNGLUSEPROGRAMPROC, glUseProgram)                     \
    X(PFNGLVIEWPORTPROC, glViewport)

#define MV_GL_DECLARE(type, name) extern type mvgl_##name;
MV_GL_FUNCTIONS(MV_GL_DECLARE)
#undef MV_GL_DECLARE

#define glActiveTexture mvgl_glActiveTexture
#define glAttachShader mvgl_glAttachShader
#define glBindFramebuffer mvgl_glBindFramebuffer
#define glBindTexture mvgl_glBindTexture
#define glBindVertexArray mvgl_glBindVertexArray
#define glBlendFunc mvgl_glBlendFunc
#define glCheckFramebufferStatus mvgl_glCheckFramebufferStatus
#define glClear mvgl_glClear
#define glClearColor mvgl_glClearColor
#define glCompileShader mvgl_glCompileShader
#define glCreateProgram mvgl_glCreateProgram
#define glCreateShader mvgl_glCreateShader
#define glDeleteFramebuffers mvgl_glDeleteFramebuffers
#define glDeleteProgram mvgl_glDeleteProgram
#define glDeleteShader mvgl_glDeleteShader
#define glDeleteTextures mvgl_glDeleteTextures
#define glDeleteVertexArrays mvgl_glDeleteVertexArrays
#define glDisable mvgl_glDisable
#define glDrawArrays mvgl_glDrawArrays
#define glEnable mvgl_glEnable
#define glFinish mvgl_glFinish
#define glFramebufferTexture2D mvgl_glFramebufferTexture2D
#define glGenFramebuffers mvgl_glGenFramebuffers
#define glGenTextures mvgl_glGenTextures
#define glGenVertexArrays mvgl_glGenVertexArrays
#define glGetError mvgl_glGetError
#define glGetIntegerv mvgl_glGetIntegerv
#define glGetProgramInfoLog mvgl_glGetProgramInfoLog
#define glGetProgramiv mvgl_glGetProgramiv
#define glGetShaderInfoLog mvgl_glGetShaderInfoLog
#define glGetShaderiv mvgl_glGetShaderiv
#define glGetString mvgl_glGetString
#define glGetUniformLocation mvgl_glGetUniformLocation
#define glLinkProgram mvgl_glLinkProgram
#define glPixelStorei mvgl_glPixelStorei
#define glReadPixels mvgl_glReadPixels
#define glScissor mvgl_glScissor
#define glShaderSource mvgl_glShaderSource
#define glTexImage2D mvgl_glTexImage2D
#define glTexImage3D mvgl_glTexImage3D
#define glTexParameteri mvgl_glTexParameteri
#define glTexSubImage2D mvgl_glTexSubImage2D
#define glUniform1f mvgl_glUniform1f
#define glUniform1i mvgl_glUniform1i
#define glUniform2f mvgl_glUniform2f
#define glUniform3f mvgl_glUniform3f
#define glUniform3fv mvgl_glUniform3fv
#define glUniformMatrix4fv mvgl_glUniformMatrix4fv
#define glUseProgram mvgl_glUseProgram
#define glViewport mvgl_glViewport

namespace mv {
using GLProcLoader = void* (*)(const char* name);
// Fills every mvgl_* pointer. Returns false and lists missing entry points if any failed.
bool loadGL(GLProcLoader loader, std::string* missing);
}  // namespace mv
#endif
