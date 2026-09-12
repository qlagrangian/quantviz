#pragma once
// viz/src/gl/gl_loader.hpp — 使う OpenGL 関数だけを自前で解決する最小ローダ（viz 層のみ）
//
// 責務: glad/GLEW のような外部ローダに依存せず、`glfwGetProcAddress` で GL 3.0 core 相当の
// 関数ポインタを 1 回だけ引く（設計書 §12.1）。GL 1.1 を超える関数は `<GL/gl.h>` には無いので、
// ここで必要な型・enum・シグネチャを自分で宣言する。システムの GL シンボルと衝突しないよう、
// ポインタ名はすべて `qv_` 接頭辞、型・定数は `quantviz::viz::glapi` 名前空間に閉じる。
//
// 使い方: `main()` が ImGui の初期化直後（= GL コンテキストが current な状態）で `gl_load()` を
// 1 回だけ呼ぶ。失敗したら false（欠けた関数名は stderr）。以後 `gl_available()` が状態を返すので、
// 3D パネルはそれを見て「OpenGL functions unavailable」表示にフォールバックできる。
//
// 注意: GLX / NSGL は GL 1.1 の関数も `glfwGetProcAddress` で返すのでここでまとめて引いているが、
// Win32 の WGL は 1.1 を返さない。Windows で viewer を動かす日が来たら、1.1 の分だけ
// `<GL/gl.h>` の静的シンボルへフォールバックすること（プラットフォーム分岐はそこだけで済む）。

#include <cstddef>
#include <cstdint>

// GL の呼び出し規約。Win32 は __stdcall、それ以外は無印。マクロはプリプロセッサの住人なので
// 名前空間の外に置く。このヘッダの宣言と gl_loader.cpp の定義の両方で使うので #undef しない。
#if defined(_WIN32)
#define QV_GLAPIENTRY __stdcall
#else
#define QV_GLAPIENTRY
#endif

namespace quantviz::viz::glapi {

// ---------------------------------------------------------------------------
// 型（<GL/gl.h> / <GL/glext.h> と同じ実体。namespace に閉じてあるので衝突しない）
// ---------------------------------------------------------------------------
using GLenum     = unsigned int;
using GLboolean  = unsigned char;
using GLbitfield = unsigned int;
using GLbyte     = signed char;
using GLubyte    = unsigned char;
using GLint      = int;
using GLuint     = unsigned int;
using GLsizei    = int;
using GLfloat    = float;
using GLchar     = char;
using GLintptr   = std::intptr_t;
using GLsizeiptr = std::intptr_t;

// ---------------------------------------------------------------------------
// enum（使うものだけ）
// ---------------------------------------------------------------------------
inline constexpr GLenum kGlNoError = 0x0000;
inline constexpr GLint  kGlFalse   = 0;
inline constexpr GLint  kGlTrue    = 1;

// バッファ
inline constexpr GLenum kGlArrayBuffer        = 0x8892;
inline constexpr GLenum kGlElementArrayBuffer = 0x8893;
inline constexpr GLenum kGlStaticDraw         = 0x88E4;
inline constexpr GLenum kGlDynamicDraw        = 0x88E8;

// 型 enum / プリミティブ
inline constexpr GLenum kGlFloat        = 0x1406;
inline constexpr GLenum kGlUnsignedInt  = 0x1405;
inline constexpr GLenum kGlUnsignedByte = 0x1401;
inline constexpr GLenum kGlTriangles    = 0x0004;

// シェーダ
inline constexpr GLenum kGlVertexShader   = 0x8B31;
inline constexpr GLenum kGlFragmentShader = 0x8B30;
inline constexpr GLenum kGlCompileStatus  = 0x8B81;
inline constexpr GLenum kGlLinkStatus     = 0x8B82;
inline constexpr GLenum kGlInfoLogLength  = 0x8B84;

// FBO / renderbuffer
inline constexpr GLenum kGlFramebuffer         = 0x8D40;
inline constexpr GLenum kGlRenderbuffer        = 0x8D41;
inline constexpr GLenum kGlColorAttachment0    = 0x8CE0;
inline constexpr GLenum kGlDepthAttachment     = 0x8D00;
inline constexpr GLenum kGlDepthComponent24    = 0x81A6;
inline constexpr GLenum kGlFramebufferComplete = 0x8CD5;
inline constexpr GLenum kGlFramebufferBinding  = 0x8CA6;
inline constexpr GLenum kGlViewportState       = 0x0BA2;

// テクスチャ
inline constexpr GLenum kGlTexture2d        = 0x0DE1;
inline constexpr GLenum kGlRgba8            = 0x8058;
inline constexpr GLenum kGlRgba             = 0x1908;
inline constexpr GLenum kGlTextureMinFilter = 0x2801;
inline constexpr GLenum kGlTextureMagFilter = 0x2800;
inline constexpr GLenum kGlTextureWrapS     = 0x2802;
inline constexpr GLenum kGlTextureWrapT     = 0x2803;
inline constexpr GLint  kGlLinear           = 0x2601;
inline constexpr GLint  kGlClampToEdge      = 0x812F;

// 固定機能の状態
inline constexpr GLenum     kGlDepthTest       = 0x0B71;
inline constexpr GLenum     kGlCullFace        = 0x0B44;
inline constexpr GLenum     kGlBlend           = 0x0BE2;
inline constexpr GLenum     kGlScissorTest     = 0x0C11;
inline constexpr GLbitfield kGlColorBufferBit  = 0x00004000;
inline constexpr GLbitfield kGlDepthBufferBit  = 0x00000100;
inline constexpr GLenum     kGlFrontAndBack    = 0x0408;
inline constexpr GLenum     kGlFill            = 0x1B02;
inline constexpr GLenum     kGlLine            = 0x1B01;

// 文字列クエリ
inline constexpr GLenum kGlVendor                = 0x1F00;
inline constexpr GLenum kGlRenderer              = 0x1F01;
inline constexpr GLenum kGlVersion               = 0x1F02;
inline constexpr GLenum kGlShadingLanguageVersion = 0x8B8C;

// ---------------------------------------------------------------------------
// 関数ポインタ（gl_load() が埋める。未ロードのまま呼んではならない）
// ---------------------------------------------------------------------------

// VAO
extern void (QV_GLAPIENTRY* qv_glGenVertexArrays)(GLsizei, GLuint*);
extern void (QV_GLAPIENTRY* qv_glBindVertexArray)(GLuint);
extern void (QV_GLAPIENTRY* qv_glDeleteVertexArrays)(GLsizei, const GLuint*);

// VBO / EBO
extern void (QV_GLAPIENTRY* qv_glGenBuffers)(GLsizei, GLuint*);
extern void (QV_GLAPIENTRY* qv_glBindBuffer)(GLenum, GLuint);
extern void (QV_GLAPIENTRY* qv_glDeleteBuffers)(GLsizei, const GLuint*);
extern void (QV_GLAPIENTRY* qv_glBufferData)(GLenum, GLsizeiptr, const void*, GLenum);
extern void (QV_GLAPIENTRY* qv_glBufferSubData)(GLenum, GLintptr, GLsizeiptr, const void*);

// 頂点属性
extern void (QV_GLAPIENTRY* qv_glVertexAttribPointer)(GLuint, GLint, GLenum, GLboolean, GLsizei,
                                                      const void*);
extern void (QV_GLAPIENTRY* qv_glEnableVertexAttribArray)(GLuint);
extern void (QV_GLAPIENTRY* qv_glBindAttribLocation)(GLuint, GLuint, const GLchar*);

// シェーダ / プログラム
extern GLuint (QV_GLAPIENTRY* qv_glCreateShader)(GLenum);
extern void (QV_GLAPIENTRY* qv_glShaderSource)(GLuint, GLsizei, const GLchar* const*, const GLint*);
extern void (QV_GLAPIENTRY* qv_glCompileShader)(GLuint);
extern void (QV_GLAPIENTRY* qv_glGetShaderiv)(GLuint, GLenum, GLint*);
extern void (QV_GLAPIENTRY* qv_glGetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
extern void (QV_GLAPIENTRY* qv_glDeleteShader)(GLuint);
extern GLuint (QV_GLAPIENTRY* qv_glCreateProgram)();
extern void (QV_GLAPIENTRY* qv_glAttachShader)(GLuint, GLuint);
extern void (QV_GLAPIENTRY* qv_glLinkProgram)(GLuint);
extern void (QV_GLAPIENTRY* qv_glGetProgramiv)(GLuint, GLenum, GLint*);
extern void (QV_GLAPIENTRY* qv_glGetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
extern void (QV_GLAPIENTRY* qv_glDeleteProgram)(GLuint);
extern void (QV_GLAPIENTRY* qv_glUseProgram)(GLuint);

// uniform
extern GLint (QV_GLAPIENTRY* qv_glGetUniformLocation)(GLuint, const GLchar*);
extern void (QV_GLAPIENTRY* qv_glUniform1f)(GLint, GLfloat);
extern void (QV_GLAPIENTRY* qv_glUniform3f)(GLint, GLfloat, GLfloat, GLfloat);
extern void (QV_GLAPIENTRY* qv_glUniformMatrix4fv)(GLint, GLsizei, GLboolean, const GLfloat*);

// FBO / renderbuffer
extern void (QV_GLAPIENTRY* qv_glGenFramebuffers)(GLsizei, GLuint*);
extern void (QV_GLAPIENTRY* qv_glBindFramebuffer)(GLenum, GLuint);
extern void (QV_GLAPIENTRY* qv_glDeleteFramebuffers)(GLsizei, const GLuint*);
extern void (QV_GLAPIENTRY* qv_glFramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint);
extern void (QV_GLAPIENTRY* qv_glFramebufferRenderbuffer)(GLenum, GLenum, GLenum, GLuint);
extern GLenum (QV_GLAPIENTRY* qv_glCheckFramebufferStatus)(GLenum);
extern void (QV_GLAPIENTRY* qv_glGenRenderbuffers)(GLsizei, GLuint*);
extern void (QV_GLAPIENTRY* qv_glBindRenderbuffer)(GLenum, GLuint);
extern void (QV_GLAPIENTRY* qv_glDeleteRenderbuffers)(GLsizei, const GLuint*);
extern void (QV_GLAPIENTRY* qv_glRenderbufferStorage)(GLenum, GLenum, GLsizei, GLsizei);

// テクスチャ
extern void (QV_GLAPIENTRY* qv_glGenTextures)(GLsizei, GLuint*);
extern void (QV_GLAPIENTRY* qv_glBindTexture)(GLenum, GLuint);
extern void (QV_GLAPIENTRY* qv_glDeleteTextures)(GLsizei, const GLuint*);
extern void (QV_GLAPIENTRY* qv_glTexImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum,
                                             GLenum, const void*);
extern void (QV_GLAPIENTRY* qv_glTexParameteri)(GLenum, GLenum, GLint);

// 固定機能の状態 / 描画
extern void (QV_GLAPIENTRY* qv_glViewport)(GLint, GLint, GLsizei, GLsizei);
extern void (QV_GLAPIENTRY* qv_glClear)(GLbitfield);
extern void (QV_GLAPIENTRY* qv_glClearColor)(GLfloat, GLfloat, GLfloat, GLfloat);
extern void (QV_GLAPIENTRY* qv_glEnable)(GLenum);
extern void (QV_GLAPIENTRY* qv_glDisable)(GLenum);
extern void (QV_GLAPIENTRY* qv_glDrawElements)(GLenum, GLsizei, GLenum, const void*);
extern void (QV_GLAPIENTRY* qv_glPolygonMode)(GLenum, GLenum);
extern GLenum (QV_GLAPIENTRY* qv_glGetError)();
extern void (QV_GLAPIENTRY* qv_glGetIntegerv)(GLenum, GLint*);
extern const GLubyte* (QV_GLAPIENTRY* qv_glGetString)(GLenum);

// ---------------------------------------------------------------------------
// ロード
// ---------------------------------------------------------------------------

/// 上のポインタをすべて `glfwGetProcAddress` で解決する。GL コンテキストが current な状態で、
/// 1 回だけ呼ぶこと（2 回目以降は前回の結果をそのまま返す）。1 つでも null なら、その名前を
/// stderr に出して false を返す（呼び出し側は 3D 描画を諦めて viewer は動かし続ける）。
bool gl_load();

/// `gl_load()` が成功したか。呼ぶ前は false。
[[nodiscard]] bool gl_available() noexcept;

/// main.cpp が GL コンテキストを作るときに選んだ GLSL のバージョン行（"#version 130" など）を
/// 控える。シェーダを書くのは viz 層の各レンダラだが、コンテキストのバージョンを決めるのは
/// main.cpp なので、両者がずれないよう出所を 1 つにする。
/// `v` は静的記憶域の文字列（リテラル）であること — ポインタをそのまま保持する。
void set_glsl_version(const char* v) noexcept;

/// `set_glsl_version()` で控えた値。未設定ならプラットフォーム既定（macOS は 150、他は 130）。
[[nodiscard]] const char* glsl_version() noexcept;

/// GL_VERSION / GL_RENDERER 等の文字列。ロード前・取得できない場合は "?"。
[[nodiscard]] const char* gl_info(GLenum name) noexcept;

}  // namespace quantviz::viz::glapi
