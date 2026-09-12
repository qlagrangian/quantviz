#pragma once
// viz/gl/math.hpp — 3D 描画のための最小限の線形代数（vizcore: GUI・GL ヘッダ非依存）。
//
// 責務: OpenGL にそのまま渡せる列優先（column-major）の 4×4 float 行列と 3 成分ベクトル、および
// look-at / perspective / 逆行列。`Mat4::m` は GL の `glUniformMatrix4fv(..., GL_FALSE, m.data())`
// と同じ並び、すなわち `m[col * 4 + row]`（列が連続）。`at(row, col)` は数学の添字で読み書きする
// ための糖衣で、格納順は変えない。規約は右手系・カメラは −z を向く・クリップ空間の z は [−1, 1]
// （OpenGL 既定。DirectX の [0, 1] ではない）。
//
// 型は一貫して `float`（GL に上げる型）。ヒープも例外も使わず、可能な限り constexpr。
// 数学関数（sqrt / tan）を使う look_at・perspective だけが非 constexpr。

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

namespace quantviz::viz::gl {

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

struct Vec3 {
    float x{}, y{}, z{};
};

constexpr Vec3 operator+(Vec3 a, Vec3 b) noexcept { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr Vec3 operator-(Vec3 a, Vec3 b) noexcept { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
constexpr Vec3 operator-(Vec3 a) noexcept { return Vec3{-a.x, -a.y, -a.z}; }
constexpr Vec3 operator*(Vec3 a, float s) noexcept { return Vec3{a.x * s, a.y * s, a.z * s}; }
constexpr Vec3 operator*(float s, Vec3 a) noexcept { return a * s; }

constexpr float dot(Vec3 a, Vec3 b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }

constexpr Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

[[nodiscard]] inline float length(Vec3 v) noexcept { return std::sqrt(dot(v, v)); }

/// 長さ 0（や NaN）のベクトルは (0,0,0) を返す — 呼び手が 0 除算を気にしなくてよいように。
[[nodiscard]] inline Vec3 normalize(Vec3 v) noexcept {
    const float len = length(v);
    return len > 0.f ? Vec3{v.x / len, v.y / len, v.z / len} : Vec3{0.f, 0.f, 0.f};
}

// ---------------------------------------------------------------------------
// Mat4 — 列優先 4×4
// ---------------------------------------------------------------------------

struct Mat4 {
    std::array<float, 16> m{};  ///< m[col * 4 + row]（OpenGL の並び）

    static constexpr Mat4 identity() noexcept {
        Mat4 r{};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.f;
        return r;
    }

    [[nodiscard]] constexpr float at(std::size_t row, std::size_t col) const noexcept {
        return m[col * 4 + row];
    }
    [[nodiscard]] constexpr float& at(std::size_t row, std::size_t col) noexcept {
        return m[col * 4 + row];
    }
};

/// 行列積（数学どおり (a·b) を点に左から掛ける = 先に b、次に a を適用）。
constexpr Mat4 operator*(const Mat4& a, const Mat4& b) noexcept {
    Mat4 r{};
    for (std::size_t c = 0; c < 4; ++c)
        for (std::size_t row = 0; row < 4; ++row) {
            float s = 0.f;
            for (std::size_t k = 0; k < 4; ++k) s += a.at(row, k) * b.at(k, c);
            r.at(row, c) = s;
        }
    return r;
}

/// 点（w = 1）を変換して同次除算する。w が 0 のとき（= カメラ平面上）は除算せずに返す。
constexpr Vec3 transform_point(const Mat4& a, Vec3 p) noexcept {
    const float x = a.at(0, 0) * p.x + a.at(0, 1) * p.y + a.at(0, 2) * p.z + a.at(0, 3);
    const float y = a.at(1, 0) * p.x + a.at(1, 1) * p.y + a.at(1, 2) * p.z + a.at(1, 3);
    const float z = a.at(2, 0) * p.x + a.at(2, 1) * p.y + a.at(2, 2) * p.z + a.at(2, 3);
    const float w = a.at(3, 0) * p.x + a.at(3, 1) * p.y + a.at(3, 2) * p.z + a.at(3, 3);
    if (w == 0.f) return Vec3{x, y, z};
    const float inv_w = 1.f / w;
    return Vec3{x * inv_w, y * inv_w, z * inv_w};
}

/// 方向ベクトル（w = 0）の変換。平行移動は効かない。
constexpr Vec3 transform_direction(const Mat4& a, Vec3 v) noexcept {
    return Vec3{a.at(0, 0) * v.x + a.at(0, 1) * v.y + a.at(0, 2) * v.z,
                a.at(1, 0) * v.x + a.at(1, 1) * v.y + a.at(1, 2) * v.z,
                a.at(2, 0) * v.x + a.at(2, 1) * v.y + a.at(2, 2) * v.z};
}

// ---------------------------------------------------------------------------
// ビュー・射影
// ---------------------------------------------------------------------------

/// gluLookAt 相当（右手系）。カメラはビュー空間で原点に置かれ、−z 方向を向く。
/// `up` が視線と平行だと退化する（`OrbitCamera` は pitch のクランプでそれを避ける）。
[[nodiscard]] inline Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) noexcept {
    const Vec3 f = normalize(target - eye);  // 前方
    const Vec3 s = normalize(cross(f, up));  // 右
    const Vec3 u = cross(s, f);              // 真上（f, s と直交するので再正規化不要）

    Mat4 r     = Mat4::identity();
    r.at(0, 0) = s.x;
    r.at(0, 1) = s.y;
    r.at(0, 2) = s.z;
    r.at(1, 0) = u.x;
    r.at(1, 1) = u.y;
    r.at(1, 2) = u.z;
    r.at(2, 0) = -f.x;
    r.at(2, 1) = -f.y;
    r.at(2, 2) = -f.z;
    r.at(0, 3) = -dot(s, eye);
    r.at(1, 3) = -dot(u, eye);
    r.at(2, 3) = dot(f, eye);
    return r;
}

/// gluPerspective 相当。クリップ空間の z は [−1, 1]（OpenGL 規約）。
/// `aspect` は幅/高さ。ウィンドウ最小化などで 0・負・NaN が来たら 1 とみなす（描画を壊さない）。
[[nodiscard]] inline Mat4 perspective(float fovy_rad, float aspect, float znear, float zfar) noexcept {
    const float a = (aspect > 0.f) ? aspect : 1.f;  // NaN は !(x > 0) で弾かれる
    const float f = 1.f / std::tan(fovy_rad * 0.5f);

    Mat4 r{};
    r.at(0, 0) = f / a;
    r.at(1, 1) = f;
    r.at(2, 2) = (zfar + znear) / (znear - zfar);
    r.at(2, 3) = (2.f * zfar * znear) / (znear - zfar);
    r.at(3, 2) = -1.f;
    return r;
}

// ---------------------------------------------------------------------------
// 逆行列
// ---------------------------------------------------------------------------

namespace detail {
constexpr bool is_finite(float v) noexcept {
    return v == v && v < std::numeric_limits<float>::infinity() &&
           v > -std::numeric_limits<float>::infinity();
}
constexpr float abs_f(float v) noexcept { return v < 0.f ? -v : v; }
}  // namespace detail

/// 一般の 4×4 逆行列（余因子展開）。行列式が 0 近傍・非有限なら nullopt。
/// 閾値 1e-20 は float の最小正規化数（~1e-38）より十分大きく、実用的なビュー・射影行列の
/// 行列式（|det| ≳ 1e-3）よりは十分小さい。
[[nodiscard]] constexpr std::optional<Mat4> inverse(const Mat4& a) noexcept {
    const std::array<float, 16>& m = a.m;
    std::array<float, 16>        inv{};

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    const float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (!detail::is_finite(det) || detail::abs_f(det) < 1e-20f) return std::nullopt;

    const float inv_det = 1.f / det;
    Mat4        r{};
    for (std::size_t i = 0; i < 16; ++i) r.m[i] = inv[i] * inv_det;
    return r;
}

}  // namespace quantviz::viz::gl
