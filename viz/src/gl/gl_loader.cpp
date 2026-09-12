#include "gl/gl_loader.hpp"

#include <cstdio>

#if defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>  // glfwGetProcAddress（+ GL 1.1 のヘッダ。ここでは型は自前のものを使う）

namespace quantviz::viz::glapi {

// ---------------------------------------------------------------------------
// 実体（宣言は gl_loader.hpp）
// ---------------------------------------------------------------------------
void (QV_GLAPIENTRY* qv_glGenVertexArrays)(GLsizei, GLuint*)          = nullptr;
void (QV_GLAPIENTRY* qv_glBindVertexArray)(GLuint)                    = nullptr;
void (QV_GLAPIENTRY* qv_glDeleteVertexArrays)(GLsizei, const GLuint*) = nullptr;

void (QV_GLAPIENTRY* qv_glGenBuffers)(GLsizei, GLuint*)                          = nullptr;
void (QV_GLAPIENTRY* qv_glBindBuffer)(GLenum, GLuint)                            = nullptr;
void (QV_GLAPIENTRY* qv_glDeleteBuffers)(GLsizei, const GLuint*)                 = nullptr;
void (QV_GLAPIENTRY* qv_glBufferData)(GLenum, GLsizeiptr, const void*, GLenum)   = nullptr;
void (QV_GLAPIENTRY* qv_glBufferSubData)(GLenum, GLintptr, GLsizeiptr, const void*) = nullptr;

void (QV_GLAPIENTRY* qv_glVertexAttribPointer)(GLuint, GLint, GLenum, GLboolean, GLsizei,
                                               const void*)            = nullptr;
void (QV_GLAPIENTRY* qv_glEnableVertexAttribArray)(GLuint)             = nullptr;
void (QV_GLAPIENTRY* qv_glBindAttribLocation)(GLuint, GLuint, const GLchar*) = nullptr;

GLuint (QV_GLAPIENTRY* qv_glCreateShader)(GLenum)                                        = nullptr;
void (QV_GLAPIENTRY* qv_glShaderSource)(GLuint, GLsizei, const GLchar* const*, const GLint*) = nullptr;
void (QV_GLAPIENTRY* qv_glCompileShader)(GLuint)                                         = nullptr;
void (QV_GLAPIENTRY* qv_glGetShaderiv)(GLuint, GLenum, GLint*)                           = nullptr;
void (QV_GLAPIENTRY* qv_glGetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*)          = nullptr;
void (QV_GLAPIENTRY* qv_glDeleteShader)(GLuint)                                          = nullptr;
GLuint (QV_GLAPIENTRY* qv_glCreateProgram)()                                             = nullptr;
void (QV_GLAPIENTRY* qv_glAttachShader)(GLuint, GLuint)                                  = nullptr;
void (QV_GLAPIENTRY* qv_glLinkProgram)(GLuint)                                           = nullptr;
void (QV_GLAPIENTRY* qv_glGetProgramiv)(GLuint, GLenum, GLint*)                          = nullptr;
void (QV_GLAPIENTRY* qv_glGetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*)         = nullptr;
void (QV_GLAPIENTRY* qv_glDeleteProgram)(GLuint)                                         = nullptr;
void (QV_GLAPIENTRY* qv_glUseProgram)(GLuint)                                            = nullptr;

GLint (QV_GLAPIENTRY* qv_glGetUniformLocation)(GLuint, const GLchar*)                = nullptr;
void (QV_GLAPIENTRY* qv_glUniform1f)(GLint, GLfloat)                                 = nullptr;
void (QV_GLAPIENTRY* qv_glUniform3f)(GLint, GLfloat, GLfloat, GLfloat)               = nullptr;
void (QV_GLAPIENTRY* qv_glUniformMatrix4fv)(GLint, GLsizei, GLboolean, const GLfloat*) = nullptr;

void (QV_GLAPIENTRY* qv_glGenFramebuffers)(GLsizei, GLuint*)                          = nullptr;
void (QV_GLAPIENTRY* qv_glBindFramebuffer)(GLenum, GLuint)                            = nullptr;
void (QV_GLAPIENTRY* qv_glDeleteFramebuffers)(GLsizei, const GLuint*)                 = nullptr;
void (QV_GLAPIENTRY* qv_glFramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint) = nullptr;
void (QV_GLAPIENTRY* qv_glFramebufferRenderbuffer)(GLenum, GLenum, GLenum, GLuint)    = nullptr;
GLenum (QV_GLAPIENTRY* qv_glCheckFramebufferStatus)(GLenum)                           = nullptr;
void (QV_GLAPIENTRY* qv_glGenRenderbuffers)(GLsizei, GLuint*)                         = nullptr;
void (QV_GLAPIENTRY* qv_glBindRenderbuffer)(GLenum, GLuint)                           = nullptr;
void (QV_GLAPIENTRY* qv_glDeleteRenderbuffers)(GLsizei, const GLuint*)                = nullptr;
void (QV_GLAPIENTRY* qv_glRenderbufferStorage)(GLenum, GLenum, GLsizei, GLsizei)      = nullptr;

void (QV_GLAPIENTRY* qv_glGenTextures)(GLsizei, GLuint*)          = nullptr;
void (QV_GLAPIENTRY* qv_glBindTexture)(GLenum, GLuint)            = nullptr;
void (QV_GLAPIENTRY* qv_glDeleteTextures)(GLsizei, const GLuint*) = nullptr;
void (QV_GLAPIENTRY* qv_glTexImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum,
                                      const void*)                = nullptr;
void (QV_GLAPIENTRY* qv_glTexParameteri)(GLenum, GLenum, GLint)   = nullptr;

void (QV_GLAPIENTRY* qv_glViewport)(GLint, GLint, GLsizei, GLsizei)      = nullptr;
void (QV_GLAPIENTRY* qv_glClear)(GLbitfield)                             = nullptr;
void (QV_GLAPIENTRY* qv_glClearColor)(GLfloat, GLfloat, GLfloat, GLfloat) = nullptr;
void (QV_GLAPIENTRY* qv_glEnable)(GLenum)                                = nullptr;
void (QV_GLAPIENTRY* qv_glDisable)(GLenum)                               = nullptr;
void (QV_GLAPIENTRY* qv_glDrawElements)(GLenum, GLsizei, GLenum, const void*) = nullptr;
void (QV_GLAPIENTRY* qv_glPolygonMode)(GLenum, GLenum)                   = nullptr;
GLenum (QV_GLAPIENTRY* qv_glGetError)()                                  = nullptr;
void (QV_GLAPIENTRY* qv_glGetIntegerv)(GLenum, GLint*)                   = nullptr;
const GLubyte* (QV_GLAPIENTRY* qv_glGetString)(GLenum)                   = nullptr;

namespace {

bool g_loaded    = false;  ///< gl_load() が成功したか
bool g_attempted = false;  ///< 1 回でも試したか（2 回目以降は結果を返すだけ）

/// main.cpp が `set_glsl_version()` で上書きするまでのプラットフォーム既定。
#if defined(__APPLE__)
const char* g_glsl_version = "#version 150";  // 3.2 core
#else
const char* g_glsl_version = "#version 130";  // 3.0
#endif

/// 1 本だけ解決する。取れなければ名前を stderr に出して ok を折る（全部試してから false を返す
/// ので、「何が足りない GL か」が 1 回の起動で全部わかる）。
template <class Fn>
void load_one(Fn& fn, const char* name, bool& ok) {
    fn = reinterpret_cast<Fn>(glfwGetProcAddress(name));
    if (fn == nullptr) {
        std::fprintf(stderr, "gl_load: missing GL function %s\n", name);
        ok = false;
    }
}

#define QV_LOAD(fn) load_one(qv_##fn, #fn, ok)

}  // namespace

bool gl_load() {
    if (g_attempted) return g_loaded;
    g_attempted = true;

    bool ok = true;
    QV_LOAD(glGenVertexArrays);
    QV_LOAD(glBindVertexArray);
    QV_LOAD(glDeleteVertexArrays);

    QV_LOAD(glGenBuffers);
    QV_LOAD(glBindBuffer);
    QV_LOAD(glDeleteBuffers);
    QV_LOAD(glBufferData);
    QV_LOAD(glBufferSubData);

    QV_LOAD(glVertexAttribPointer);
    QV_LOAD(glEnableVertexAttribArray);
    QV_LOAD(glBindAttribLocation);

    QV_LOAD(glCreateShader);
    QV_LOAD(glShaderSource);
    QV_LOAD(glCompileShader);
    QV_LOAD(glGetShaderiv);
    QV_LOAD(glGetShaderInfoLog);
    QV_LOAD(glDeleteShader);
    QV_LOAD(glCreateProgram);
    QV_LOAD(glAttachShader);
    QV_LOAD(glLinkProgram);
    QV_LOAD(glGetProgramiv);
    QV_LOAD(glGetProgramInfoLog);
    QV_LOAD(glDeleteProgram);
    QV_LOAD(glUseProgram);

    QV_LOAD(glGetUniformLocation);
    QV_LOAD(glUniform1f);
    QV_LOAD(glUniform3f);
    QV_LOAD(glUniformMatrix4fv);

    QV_LOAD(glGenFramebuffers);
    QV_LOAD(glBindFramebuffer);
    QV_LOAD(glDeleteFramebuffers);
    QV_LOAD(glFramebufferTexture2D);
    QV_LOAD(glFramebufferRenderbuffer);
    QV_LOAD(glCheckFramebufferStatus);
    QV_LOAD(glGenRenderbuffers);
    QV_LOAD(glBindRenderbuffer);
    QV_LOAD(glDeleteRenderbuffers);
    QV_LOAD(glRenderbufferStorage);

    QV_LOAD(glGenTextures);
    QV_LOAD(glBindTexture);
    QV_LOAD(glDeleteTextures);
    QV_LOAD(glTexImage2D);
    QV_LOAD(glTexParameteri);

    QV_LOAD(glViewport);
    QV_LOAD(glClear);
    QV_LOAD(glClearColor);
    QV_LOAD(glEnable);
    QV_LOAD(glDisable);
    QV_LOAD(glDrawElements);
    QV_LOAD(glPolygonMode);
    QV_LOAD(glGetError);
    QV_LOAD(glGetIntegerv);
    QV_LOAD(glGetString);

    g_loaded = ok;
    if (ok) {
        std::fprintf(stderr, "gl_load: OK  GL_VERSION=%s | GL_RENDERER=%s | GLSL=%s\n",
                     gl_info(kGlVersion), gl_info(kGlRenderer), gl_info(kGlShadingLanguageVersion));
    }
    return ok;
}

#undef QV_LOAD

bool gl_available() noexcept { return g_loaded; }

void set_glsl_version(const char* v) noexcept {
    if (v != nullptr && *v != '\0') g_glsl_version = v;
}

const char* glsl_version() noexcept { return g_glsl_version; }

const char* gl_info(GLenum name) noexcept {
    if (qv_glGetString == nullptr) return "?";
    const GLubyte* s = qv_glGetString(name);
    return s != nullptr ? reinterpret_cast<const char*>(s) : "?";
}

}  // namespace quantviz::viz::glapi
