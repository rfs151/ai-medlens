#include "gl_objects.h"

#include <vector>

namespace mv {
namespace {

GLuint compile(GLenum type, const char* body, std::string* log) {
    const char* sources[] = {MV_GLSL_HEADER, body};
    GLuint s = glCreateShader(type);
    glShaderSource(s, 2, sources, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> buf(size_t(len) + 1, '\0');
        glGetShaderInfoLog(s, len, nullptr, buf.data());
        if (log) *log += (type == GL_VERTEX_SHADER ? "vertex: " : "fragment: ") + std::string(buf.data());
        glDeleteShader(s);
        return 0;
    }
    return s;
}

}  // namespace

Program Program::build(const char* vertexSrc, const char* fragmentSrc, std::string* log) {
    Program p;
    GLuint vs = compile(GL_VERTEX_SHADER, vertexSrc, log);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fragmentSrc, log);
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return p;
    }
    GLuint id = glCreateProgram();
    glAttachShader(id, vs);
    glAttachShader(id, fs);
    glLinkProgram(id);
    glDeleteShader(vs);  // flagged for deletion; freed with the program
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(id, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(id, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> buf(size_t(len) + 1, '\0');
        glGetProgramInfoLog(id, len, nullptr, buf.data());
        if (log) *log += "link: " + std::string(buf.data());
        glDeleteProgram(id);
        return p;
    }
    p.id_.reset(id);
    return p;
}

bool RenderTarget::ensure(int width, int height) {
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    if (fbo_ && width == w_ && height == h_) return true;
    GLuint t = 0, f = 0;
    glGenTextures(1, &t);
    tex_.reset(t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &f);
    fbo_.reset(f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
    bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    w_ = width;
    h_ = height;
    return ok;
}

void RenderTarget::bind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_.get());
    glViewport(0, 0, w_, h_);
}

}  // namespace mv
