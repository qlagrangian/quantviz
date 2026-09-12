#include "gl/surface_renderer.hpp"

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "gl/gl_loader.hpp"

namespace quantviz::viz {

using namespace quantviz::viz::glapi;  // qv_gl* と kGl* をそのまま使う（このファイル限定）

namespace {

/// 頂点シェーダ。
///
/// 法線は「モデル行列の逆転置」で変換する必要がある（モデル行列は軸の入れ替え + 非一様スケール
/// なので、単に M を掛けると法線が傾く）。逆転置は列ベクトルの外積で厳密に作れる:
/// 列 c0,c1,c2 の行列 M に対して M⁻¹ の**行**が cross(c1,c2)/det, cross(c2,c0)/det, cross(c0,c1)/det、
/// つまり M⁻ᵀ の**列**がそれ。したがって mat3(cross(c1,c2), cross(c2,c0), cross(c0,c1)) = det(M)·M⁻ᵀ。
/// ここでのモデル行列は y と z を入れ替えるので **det < 0**（鏡像）。正規化では符号が戻らないため、
/// sign(det) を掛けて外向きの法線に直す（掛けないと法線が裏返り、フラグメント側の abs() が
/// たまたまそれを隠すだけ、という気持ち悪い状態になる）。
constexpr const char* kVertexBody = R"(
in vec3 a_pos;
in vec3 a_normal;

uniform mat4  u_mvp;
uniform mat4  u_model;
uniform float u_z_min;
uniform float u_z_max;

out vec3  v_normal;
out float v_h;      // 高さの正規化値 [0, 1]（色と等高線に使う）

void main() {
    vec3 c0 = u_model[0].xyz;
    vec3 c1 = u_model[1].xyz;
    vec3 c2 = u_model[2].xyz;
    vec3 a0 = cross(c1, c2);
    mat3 nmat = mat3(a0, cross(c2, c0), cross(c0, c1));   // det(M) * M^-T
    float det = dot(c0, a0);                              // y/z 入れ替えのぶん負になる
    v_normal = nmat * a_normal * sign(det);               // 鏡像ぶんの符号を戻す

    v_h = (a_pos.z - u_z_min) / max(u_z_max - u_z_min, 1e-6);
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
)";

/// フラグメントシェーダ。高さ色（viridis 6 段の折れ線近似）× Lambert（環境光 0.35）+ 等高線。
/// 等高線は `fract(h * lines)` の 0 近傍を `fwidth` で 1 px 幅に均す（画面解像度に依らない細線）。
constexpr const char* kFragmentBody = R"(
in vec3  v_normal;
in float v_h;

uniform vec3  u_light_dir;
uniform float u_contour_lines;

out vec4 o_color;

// matplotlib viridis の 6 標本（#440154 #414487 #2a788e #22a884 #7ad151 #fde725）を線形補間。
vec3 viridis(float t) {
    const vec3 c0 = vec3(0.267, 0.005, 0.329);
    const vec3 c1 = vec3(0.255, 0.267, 0.529);
    const vec3 c2 = vec3(0.165, 0.471, 0.558);
    const vec3 c3 = vec3(0.133, 0.659, 0.518);
    const vec3 c4 = vec3(0.478, 0.821, 0.318);
    const vec3 c5 = vec3(0.993, 0.906, 0.145);
    float u = clamp(t, 0.0, 1.0) * 5.0;
    if (u < 1.0) return mix(c0, c1, u);
    if (u < 2.0) return mix(c1, c2, u - 1.0);
    if (u < 3.0) return mix(c2, c3, u - 2.0);
    if (u < 4.0) return mix(c3, c4, u - 3.0);
    return mix(c4, c5, u - 4.0);
}

void main() {
    float t = clamp(v_h, 0.0, 1.0);
    vec3  base = viridis(t);

    // abs() で両面ライティング: 谷を下から覗いたときに面が真っ黒に潰れない。
    vec3  n   = normalize(v_normal);
    float lam = abs(dot(n, normalize(u_light_dir)));
    vec3  col = base * (0.35 + 0.65 * lam);

    if (u_contour_lines >= 1.0) {
        float s    = t * u_contour_lines;
        float dist = 0.5 - abs(fract(s) - 0.5);      // 最寄りの等高線までの距離
        float w    = max(fwidth(s), 1e-5);           // 1 px ぶんの s
        float line = 1.0 - smoothstep(0.0, w, dist);
        col = mix(col, vec3(0.04, 0.04, 0.06), line * 0.75);
    }

    o_color = vec4(col, 1.0);
}
)";

/// 幅の半分。0・負・NaN は 1 にする（軸が退化していても行列が壊れない）。
float half_or_one(float span) noexcept { return (span > 0.f) ? 0.5f * span : 1.f; }

/// バイトオフセットを glVertexAttribPointer の `pointer` 引数に渡す形へ。
const void* byte_offset(std::size_t off) noexcept {
    return reinterpret_cast<const void*>(static_cast<std::uintptr_t>(off));
}

/// シェーダ 1 本をコンパイルする。失敗したら 0 を返し、log にコンパイルログを入れる。
unsigned int compile(GLenum stage, const std::string& src, std::string& log) {
    const GLuint sh = qv_glCreateShader(stage);
    if (sh == 0) {
        log = "glCreateShader failed";
        return 0;
    }
    const GLchar* ptr = src.c_str();
    const GLint   len = static_cast<GLint>(src.size());
    qv_glShaderSource(sh, 1, &ptr, &len);
    qv_glCompileShader(sh);

    GLint ok = 0;
    qv_glGetShaderiv(sh, kGlCompileStatus, &ok);
    if (ok == kGlTrue) return sh;

    GLint n = 0;
    qv_glGetShaderiv(sh, kGlInfoLogLength, &n);
    std::vector<char> buf(static_cast<std::size_t>(n > 1 ? n : 1), '\0');
    qv_glGetShaderInfoLog(sh, static_cast<GLsizei>(buf.size()), nullptr, buf.data());
    log = buf.data();
    qv_glDeleteShader(sh);
    return 0;
}

}  // namespace

SurfaceRenderer::~SurfaceRenderer() { destroy(); }

// ---------------------------------------------------------------------------
// 初期化
// ---------------------------------------------------------------------------
bool SurfaceRenderer::init(std::size_t n_x, std::size_t n_y, const char* glsl_version) {
    if (ready()) return true;
    error_.clear();  // 前回の失敗理由を持ち越さない（今回の成否だけを error_ に残す）
    if (!gl_available()) {
        error_ = "OpenGL functions are not loaded (gl_load() failed)";
        return false;
    }
    if (n_x < 2 || n_y < 2) {
        error_ = "SurfaceRenderer::init needs at least a 2x2 grid";
        return false;
    }
    n_x_ = n_x;
    n_y_ = n_y;

    if (!build_program(glsl_version != nullptr ? glsl_version : glapi::glsl_version())) {
        destroy();
        return false;
    }

    // ---- VAO / VBO / EBO。容量はここで 1 回だけ確保する（以後 glBufferSubData で上書き）。
    //
    // GL_ELEMENT_ARRAY_BUFFER の束縛は **VAO の状態**なので、core profile では VAO 0（= 存在しない）
    // のもとで触ると GL_INVALID_OPERATION になる。EBO の操作は必ず vao_ を束縛した中で行い、
    // VAO の外で GL_ELEMENT_ARRAY_BUFFER を 0 に戻したりしない（VAO を外せば一緒に外れる）。
    const std::size_t vertex_bytes = n_x_ * n_y_ * sizeof(gl::SurfaceMesh::Vertex);
    const std::size_t index_bytes  = 6 * (n_x_ - 1) * (n_y_ - 1) * sizeof(std::uint32_t);

    qv_glGenVertexArrays(1, &vao_);
    qv_glBindVertexArray(vao_);

    qv_glGenBuffers(1, &vbo_);
    qv_glBindBuffer(kGlArrayBuffer, vbo_);
    qv_glBufferData(kGlArrayBuffer, static_cast<GLsizeiptr>(vertex_bytes), nullptr, kGlDynamicDraw);

    constexpr GLsizei kStride = static_cast<GLsizei>(sizeof(gl::SurfaceMesh::Vertex));
    qv_glVertexAttribPointer(0, 3, kGlFloat, static_cast<GLboolean>(kGlFalse), kStride, byte_offset(0));
    qv_glEnableVertexAttribArray(0);
    qv_glVertexAttribPointer(1, 3, kGlFloat, static_cast<GLboolean>(kGlFalse), kStride,
                             byte_offset(3 * sizeof(float)));
    qv_glEnableVertexAttribArray(1);

    qv_glGenBuffers(1, &ebo_);
    qv_glBindBuffer(kGlElementArrayBuffer, ebo_);
    qv_glBufferData(kGlElementArrayBuffer, static_cast<GLsizeiptr>(index_bytes), nullptr, kGlStaticDraw);

    qv_glBindVertexArray(0);           // EBO の束縛はこの VAO と一緒に外れる
    qv_glBindBuffer(kGlArrayBuffer, 0);  // ARRAY_BUFFER は VAO の状態ではないので外してよい

    // ---- FBO（カラーテクスチャ + 深度 renderbuffer）。実サイズは最初の resize() で決まる。
    qv_glGenFramebuffers(1, &fbo_);
    qv_glGenTextures(1, &color_tex_);
    qv_glGenRenderbuffers(1, &depth_rb_);
    resize(640, 480);

    if (!fbo_ok_) {
        destroy();
        return false;
    }
    return true;
}

bool SurfaceRenderer::build_program(const char* glsl_version) {
    const std::string head = std::string(glsl_version) + "\n";
    std::string       log;

    const GLuint vs = compile(kGlVertexShader, head + kVertexBody, log);
    if (vs == 0) {
        error_ = "vertex shader: " + log;
        return false;
    }
    const GLuint fs = compile(kGlFragmentShader, head + kFragmentBody, log);
    if (fs == 0) {
        error_ = "fragment shader: " + log;
        qv_glDeleteShader(vs);
        return false;
    }

    program_ = qv_glCreateProgram();
    qv_glAttachShader(program_, vs);
    qv_glAttachShader(program_, fs);
    // 属性位置はリンク前に固定する（location 修飾子は GLSL 330 からなので 130 では使えない）。
    qv_glBindAttribLocation(program_, 0, "a_pos");
    qv_glBindAttribLocation(program_, 1, "a_normal");
    qv_glLinkProgram(program_);

    GLint ok = 0;
    qv_glGetProgramiv(program_, kGlLinkStatus, &ok);
    if (ok != kGlTrue) {
        GLint n = 0;
        qv_glGetProgramiv(program_, kGlInfoLogLength, &n);
        std::vector<char> buf(static_cast<std::size_t>(n > 1 ? n : 1), '\0');
        qv_glGetProgramInfoLog(program_, static_cast<GLsizei>(buf.size()), nullptr, buf.data());
        error_ = std::string("program link: ") + buf.data();
        qv_glDeleteProgram(program_);
        program_ = 0;
    }
    qv_glDeleteShader(vs);  // プログラムにアタッチ済みなので、ここで参照を落としてよい
    qv_glDeleteShader(fs);
    if (program_ == 0) return false;

    u_mvp_     = qv_glGetUniformLocation(program_, "u_mvp");
    u_model_   = qv_glGetUniformLocation(program_, "u_model");
    u_light_   = qv_glGetUniformLocation(program_, "u_light_dir");
    u_z_min_   = qv_glGetUniformLocation(program_, "u_z_min");
    u_z_max_   = qv_glGetUniformLocation(program_, "u_z_max");
    u_contour_ = qv_glGetUniformLocation(program_, "u_contour_lines");
    return true;
}

// ---------------------------------------------------------------------------
// 頂点の更新
// ---------------------------------------------------------------------------
void SurfaceRenderer::upload(const gl::SurfaceMesh& mesh) {
    if (!ready()) return;
    if (mesh.n_x() != n_x_ || mesh.n_y() != n_y_) return;  // 格子が違うメッシュは受け取らない

    const std::span<const gl::SurfaceMesh::Vertex> v = mesh.vertices();

    // x/y の範囲（モデル行列の正規化に使う）。`SurfaceMesh` は v.x = xs[ix]、v.y = ys[iy] なので、
    // 0 行目に x の全値が、0 列目に y の全値が漏れなく現れる。全頂点を舐める必要はなく
    // O(n_x + n_y) で厳密に求まる（200×200 なら 40000 → 400 回）。軸の単調性も仮定しない。
    x_min_ = x_max_ = v[0].x;
    for (std::size_t ix = 1; ix < n_x_; ++ix) {
        const float x = v[ix].x;
        x_min_        = x < x_min_ ? x : x_min_;
        x_max_        = x > x_max_ ? x : x_max_;
    }
    y_min_ = y_max_ = v[0].y;
    for (std::size_t iy = 1; iy < n_y_; ++iy) {
        const float y = v[iy * n_x_].y;
        y_min_        = y < y_min_ ? y : y_min_;
        y_max_        = y > y_max_ ? y : y_max_;
    }

    qv_glBindBuffer(kGlArrayBuffer, vbo_);
    qv_glBufferSubData(kGlArrayBuffer, 0,
                       static_cast<GLsizeiptr>(v.size() * sizeof(gl::SurfaceMesh::Vertex)), v.data());
    qv_glBindBuffer(kGlArrayBuffer, 0);

    if (index_count_ == 0) {  // インデックスは格子サイズだけで決まる → 最初の 1 回きり
        const std::span<const std::uint32_t> idx = mesh.indices();
        // EBO は VAO の状態。core profile で怒られないよう vao_ を束縛した中だけで触る。
        qv_glBindVertexArray(vao_);
        qv_glBufferSubData(kGlElementArrayBuffer, 0,
                           static_cast<GLsizeiptr>(idx.size() * sizeof(std::uint32_t)), idx.data());
        qv_glBindVertexArray(0);
        index_count_ = static_cast<int>(idx.size());
    }
}

// ---------------------------------------------------------------------------
// 描画先のサイズ
// ---------------------------------------------------------------------------
void SurfaceRenderer::resize(int w, int h) {
    if (fbo_ == 0) return;
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    if (w == fb_w_ && h == fb_h_) return;
    fb_w_ = w;
    fb_h_ = h;

    qv_glBindTexture(kGlTexture2d, color_tex_);
    qv_glTexImage2D(kGlTexture2d, 0, static_cast<GLint>(kGlRgba8), w, h, 0, kGlRgba, kGlUnsignedByte,
                    nullptr);
    qv_glTexParameteri(kGlTexture2d, kGlTextureMinFilter, kGlLinear);
    qv_glTexParameteri(kGlTexture2d, kGlTextureMagFilter, kGlLinear);
    qv_glTexParameteri(kGlTexture2d, kGlTextureWrapS, kGlClampToEdge);
    qv_glTexParameteri(kGlTexture2d, kGlTextureWrapT, kGlClampToEdge);
    qv_glBindTexture(kGlTexture2d, 0);

    qv_glBindRenderbuffer(kGlRenderbuffer, depth_rb_);
    qv_glRenderbufferStorage(kGlRenderbuffer, kGlDepthComponent24, w, h);
    qv_glBindRenderbuffer(kGlRenderbuffer, 0);

    GLint prev_fbo = 0;
    qv_glGetIntegerv(kGlFramebufferBinding, &prev_fbo);
    qv_glBindFramebuffer(kGlFramebuffer, fbo_);
    qv_glFramebufferTexture2D(kGlFramebuffer, kGlColorAttachment0, kGlTexture2d, color_tex_, 0);
    qv_glFramebufferRenderbuffer(kGlFramebuffer, kGlDepthAttachment, kGlRenderbuffer, depth_rb_);
    const GLenum status = qv_glCheckFramebufferStatus(kGlFramebuffer);
    qv_glBindFramebuffer(kGlFramebuffer, static_cast<GLuint>(prev_fbo));

    fbo_ok_ = (status == kGlFramebufferComplete);
    if (!fbo_ok_) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "framebuffer incomplete (0x%X)", status);
        error_ = buf;
    }
}

// ---------------------------------------------------------------------------
// 描画
// ---------------------------------------------------------------------------
void SurfaceRenderer::render(const gl::OrbitCamera& cam, float z_min, float z_max, int contour_lines,
                             bool wire) {
    if (!ready() || !fbo_ok_ || fb_w_ < 1 || fb_h_ < 1) return;

    // ImGui のバックエンドは「既定 FBO・自前のビューポート・深度オフ・ポリゴン塗りつぶし」を
    // 前提にするので、最後に必ずそこへ戻す。
    GLint prev_fbo = 0;
    GLint prev_vp[4]{0, 0, 0, 0};
    qv_glGetIntegerv(kGlFramebufferBinding, &prev_fbo);
    qv_glGetIntegerv(kGlViewportState, prev_vp);

    qv_glBindFramebuffer(kGlFramebuffer, fbo_);
    qv_glViewport(0, 0, fb_w_, fb_h_);
    qv_glEnable(kGlDepthTest);
    qv_glDisable(kGlCullFace);  // 面は両側から見える（下から覗いても穴が空かない）
    qv_glDisable(kGlBlend);
    qv_glDisable(kGlScissorTest);
    qv_glClearColor(0.10f, 0.11f, 0.13f, 1.0f);
    qv_glClear(kGlColorBufferBit | kGlDepthBufferBit);

    // クリアより後で降りること: まだ頂点が無い（= 初回 / リサイズ直後）フレームでも、
    // テクスチャは「背景色で塗られた空の視点」になる。ここで早期 return すると、
    // glTexImage2D(nullptr) で確保しただけの未初期化テクスチャがそのまま表示されてしまう。
    if (index_count_ > 0) {
        // ---- 行列。モデル行列が (x, y, 高さ) → (x', 高さ', y') の軸入れ替え + 正規化を担う。
        const float xh = half_or_one(x_max_ - x_min_);
        const float xm = 0.5f * (x_min_ + x_max_);
        const float yh = half_or_one(y_max_ - y_min_);
        const float ym = 0.5f * (y_min_ + y_max_);
        const float zh = half_or_one(z_max - z_min);
        const float zm = 0.5f * (z_min + z_max);

        gl::Mat4 model{};
        model.at(0, 0) = 1.f / xh;
        model.at(0, 3) = -xm / xh;
        model.at(1, 2) = kHeight / zh;
        model.at(1, 3) = -kHeight * zm / zh;
        model.at(2, 1) = 1.f / yh;
        model.at(2, 3) = -ym / yh;
        model.at(3, 3) = 1.f;

        const float    aspect = static_cast<float>(fb_w_) / static_cast<float>(fb_h_);
        const gl::Mat4 mvp    = cam.projection(aspect) * cam.view() * model;

        qv_glUseProgram(program_);
        constexpr GLboolean kNoTranspose = static_cast<GLboolean>(kGlFalse);
        qv_glUniformMatrix4fv(u_mvp_, 1, kNoTranspose, mvp.m.data());
        qv_glUniformMatrix4fv(u_model_, 1, kNoTranspose, model.m.data());
        qv_glUniform3f(u_light_, 0.40f, 0.82f, 0.41f);  // ワールド固定の平行光（正規化はシェーダ側）
        qv_glUniform1f(u_z_min_, z_min);
        qv_glUniform1f(u_z_max_, z_max);
        qv_glUniform1f(u_contour_, static_cast<GLfloat>(contour_lines > 0 ? contour_lines : 0));

        if (wire) qv_glPolygonMode(kGlFrontAndBack, kGlLine);
        qv_glBindVertexArray(vao_);
        qv_glDrawElements(kGlTriangles, index_count_, kGlUnsignedInt, nullptr);
        qv_glBindVertexArray(0);
        if (wire) qv_glPolygonMode(kGlFrontAndBack, kGlFill);  // ImGui はポリゴン塗りつぶし前提

        qv_glUseProgram(0);
    }

    qv_glDisable(kGlDepthTest);
    qv_glBindFramebuffer(kGlFramebuffer, static_cast<GLuint>(prev_fbo));
    qv_glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3]);

    // デバッグ: GL エラーは 1 回だけ報告する（毎フレーム出すとログが埋まる）。
    const GLenum err = qv_glGetError();
    if (err != kGlNoError && !error_logged_) {
        std::fprintf(stderr, "SurfaceRenderer::render: glGetError = 0x%X\n", err);
        error_logged_ = true;
    }
}

// ---------------------------------------------------------------------------
// 破棄
// ---------------------------------------------------------------------------
void SurfaceRenderer::destroy() {
    // GL コンテキストが current な状態で呼ばれることが前提（デストラクタ経由も同じ）。
    // それを保証しているのは main.cpp 側の寿命管理で、SceneRegistry（= パネル = このレンダラ）を
    // ImGui / GLFW のシャットダウンより前のスコープで畳んでいる。下の `gl_available()` は
    // 「gl_load() が失敗して何も作られなかった」場合に空振りするためのもので、
    // **死んだコンテキストからは守れない**（守れるのは上の寿命管理だけ）。
    if (!gl_available()) return;
    if (depth_rb_ != 0) qv_glDeleteRenderbuffers(1, &depth_rb_);
    if (color_tex_ != 0) qv_glDeleteTextures(1, &color_tex_);
    if (fbo_ != 0) qv_glDeleteFramebuffers(1, &fbo_);
    if (ebo_ != 0) qv_glDeleteBuffers(1, &ebo_);
    if (vbo_ != 0) qv_glDeleteBuffers(1, &vbo_);
    if (vao_ != 0) qv_glDeleteVertexArrays(1, &vao_);
    if (program_ != 0) qv_glDeleteProgram(program_);
    depth_rb_ = color_tex_ = fbo_ = ebo_ = vbo_ = vao_ = program_ = 0;
    fb_w_ = fb_h_ = 0;
    index_count_  = 0;
    fbo_ok_       = false;
}

}  // namespace quantviz::viz
