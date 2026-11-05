#pragma once
// RAII wrappers for GL objects: move-only, delete on destruction, no hidden global state.

#include <string>
#include <utility>

#include "gl_api.h"

namespace mv {

// Move-only owner of one GL name; Deleter is called with the name when it goes out of scope.
template <void (*Deleter)(GLuint)>
class GLHandle {
public:
    GLHandle() = default;
    explicit GLHandle(GLuint id) : id_(id) {}
    ~GLHandle() { reset(); }
    GLHandle(const GLHandle&) = delete;
    GLHandle& operator=(const GLHandle&) = delete;
    GLHandle(GLHandle&& o) noexcept : id_(std::exchange(o.id_, 0)) {}
    GLHandle& operator=(GLHandle&& o) noexcept {
        if (this != &o) {
            reset();
            id_ = std::exchange(o.id_, 0);
        }
        return *this;
    }
    void reset(GLuint id = 0) {
        if (id_) Deleter(id_);
        id_ = id;
    }
    GLuint get() const { return id_; }
    explicit operator bool() const { return id_ != 0; }

private:
    GLuint id_ = 0;
};

namespace detail {
inline void deleteTexture(GLuint id) { glDeleteTextures(1, &id); }
inline void deleteFramebuffer(GLuint id) { glDeleteFramebuffers(1, &id); }
inline void deleteVertexArray(GLuint id) { glDeleteVertexArrays(1, &id); }
inline void deleteProgram(GLuint id) { glDeleteProgram(id); }
}  // namespace detail

using Texture = GLHandle<detail::deleteTexture>;
using Framebuffer = GLHandle<detail::deleteFramebuffer>;
using VertexArray = GLHandle<detail::deleteVertexArray>;

// Linked shader program with cached uniform lookups by name.
class Program {
public:
    // Compiles vertex + fragment source (MV_GLSL_HEADER is prepended); on failure returns an empty
    // program and writes the compiler log to *log.
    static Program build(const char* vertexSrc, const char* fragmentSrc, std::string* log);

    void use() const { glUseProgram(id_.get()); }
    GLint loc(const char* name) const { return glGetUniformLocation(id_.get(), name); }
    void set(const char* name, int v) const { glUniform1i(loc(name), v); }
    void set(const char* name, float v) const { glUniform1f(loc(name), v); }
    void set(const char* name, float a, float b) const { glUniform2f(loc(name), a, b); }
    void set(const char* name, float a, float b, float c) const { glUniform3f(loc(name), a, b, c); }
    void setMat4(const char* name, const float* m) const { glUniformMatrix4fv(loc(name), 1, GL_FALSE, m); }
    explicit operator bool() const { return bool(id_); }

private:
    GLHandle<detail::deleteProgram> id_;
};

// Color render target (one RGBA8 texture), resized lazily; used for reduced-resolution rendering.
class RenderTarget {
public:
    // Returns false if the framebuffer is incomplete.
    bool ensure(int width, int height);
    void bind() const;
    GLuint texture() const { return tex_.get(); }
    GLuint framebuffer() const { return fbo_.get(); }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    Texture tex_;
    Framebuffer fbo_;
    int w_ = 0, h_ = 0;
};

}  // namespace mv
