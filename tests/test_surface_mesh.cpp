// SURF-xx — viz/gl/surface_mesh.hpp（N×M 格子 → 頂点・法線・インデックス、CPU 側）の仕様テスト
//
// 格子の走査順は y メジャー行（index = iy·n_x + ix）。法線は中心差分（端は片側差分）で
// (-∂z/∂x, -∂z/∂y, 1) を正規化したもの。z の更新でバッファが作り直されないこと（= GL への
// glBufferSubData が常に同じ VBO へ書けること）も仕様の一部。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "quantviz/viz/gl/surface_mesh.hpp"

using Catch::Matchers::WithinAbs;
using quantviz::viz::gl::SurfaceMesh;

namespace {

/// [lo, hi] の等間隔軸を n 点で作る。
std::vector<float> linspace(float lo, float hi, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = lo + (hi - lo) * static_cast<float>(i) / static_cast<float>(n - 1);
    return v;
}

/// 決定的な「ランダムっぽい」高さ（seed から splitmix 風のハッシュ）。
float pseudo_z(std::size_t i) {
    std::uint64_t x = i * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    return static_cast<float>(static_cast<double>(x >> 40) / 16777216.0 - 0.5);  // [-0.5, 0.5)
}

}  // namespace

TEST_CASE("SURF-01: an n_x by n_y grid has n_x*n_y vertices and 2(n_x-1)(n_y-1) triangles",
          "[surf][unit]") {
    struct Case {
        std::size_t nx, ny;
    };
    for (const Case c : {Case{2, 2}, Case{3, 5}, Case{17, 4}, Case{64, 48}}) {
        SurfaceMesh mesh(c.nx, c.ny);
        INFO(c.nx << " x " << c.ny);
        CHECK(mesh.n_x() == c.nx);
        CHECK(mesh.n_y() == c.ny);
        CHECK(mesh.vertices().size() == c.nx * c.ny);
        CHECK(mesh.triangle_count() == 2 * (c.nx - 1) * (c.ny - 1));
        CHECK(mesh.indices().size() == 6 * (c.nx - 1) * (c.ny - 1));
    }
}

TEST_CASE("SURF-02: every index is below the vertex count and every triangle winds CCW",
          "[surf][property]") {
    constexpr std::size_t kNx = 13, kNy = 9;
    SurfaceMesh           mesh(kNx, kNy);
    mesh.set_axes(linspace(-1.f, 1.f, kNx), linspace(0.f, 2.f, kNy));

    std::vector<float> z(kNx * kNy);
    for (std::size_t i = 0; i < z.size(); ++i) z[i] = pseudo_z(i);
    mesh.set_z(z);

    const auto idx  = mesh.indices();
    const auto vert = mesh.vertices();
    REQUIRE(vert.size() == kNx * kNy);
    REQUIRE(idx.size() == 6 * (kNx - 1) * (kNy - 1));
    for (const std::uint32_t i : idx) CHECK(static_cast<std::size_t>(i) < vert.size());

    // xy 平面へ投影した三角形の符号付き面積が正 ⇒ +z から見て反時計回り（front face）。
    for (std::size_t t = 0; t < idx.size(); t += 3) {
        const auto& a = vert[idx[t]];
        const auto& b = vert[idx[t + 1]];
        const auto& c = vert[idx[t + 2]];
        const float cross_z = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        INFO("triangle " << t / 3);
        CHECK(cross_z > 0.f);
    }
}

TEST_CASE("SURF-03: every normal is unit length (1e-6)", "[surf][property]") {
    constexpr std::size_t kNx = 31, kNy = 23;
    SurfaceMesh           mesh(kNx, kNy);
    mesh.set_axes(linspace(-2.f, 2.f, kNx), linspace(-1.f, 3.f, kNy));
    REQUIRE(mesh.vertices().size() == kNx * kNy);

    std::vector<float> z(kNx * kNy);
    // なだらかな面、急峻な面、ほぼ平坦な面のいずれでも単位長であること。
    for (const float amp : {0.0f, 0.05f, 1.0f, 50.0f}) {
        for (std::size_t i = 0; i < z.size(); ++i) z[i] = amp * pseudo_z(i);
        mesh.set_z(z);
        mesh.update_normals();
        for (const auto& v : mesh.vertices()) {
            const double n = std::sqrt(static_cast<double>(v.nx) * static_cast<double>(v.nx) +
                                       static_cast<double>(v.ny) * static_cast<double>(v.ny) +
                                       static_cast<double>(v.nz) * static_cast<double>(v.nz));
            INFO("amp " << amp);
            CHECK_THAT(n, WithinAbs(1.0, 1e-6));
        }
    }
}

TEST_CASE("SURF-04: a flat plane z = const has every normal equal to (0,0,1)", "[surf][numeric]") {
    constexpr std::size_t kNx = 9, kNy = 7;
    SurfaceMesh           mesh(kNx, kNy);
    mesh.set_axes(linspace(-3.f, 4.f, kNx), linspace(10.f, 20.f, kNy));

    const std::vector<float> z(kNx * kNy, 2.5f);
    mesh.set_z(z);
    mesh.update_normals();
    REQUIRE(mesh.vertices().size() == kNx * kNy);

    for (const auto& v : mesh.vertices()) {
        CHECK_THAT(static_cast<double>(v.nx), WithinAbs(0.0, 1e-7));
        CHECK_THAT(static_cast<double>(v.ny), WithinAbs(0.0, 1e-7));
        CHECK_THAT(static_cast<double>(v.nz), WithinAbs(1.0, 1e-7));
        CHECK(v.z == 2.5f);
    }
}

TEST_CASE("SURF-05: set_z does not reallocate — buffer pointers and sizes are stable",
          "[surf][unit]") {
    constexpr std::size_t kNx = 40, kNy = 25;
    SurfaceMesh           mesh(kNx, kNy);
    mesh.set_axes(linspace(0.f, 1.f, kNx), linspace(0.f, 1.f, kNy));

    const void*       v0  = static_cast<const void*>(mesh.vertices().data());
    const void*       i0  = static_cast<const void*>(mesh.indices().data());
    const std::size_t vn0 = mesh.vertices().size();
    const std::size_t in0 = mesh.indices().size();
    REQUIRE(v0 != nullptr);
    REQUIRE(i0 != nullptr);
    REQUIRE(vn0 == kNx * kNy);
    REQUIRE(in0 == 6 * (kNx - 1) * (kNy - 1));

    std::vector<float> z(kNx * kNy);
    for (int pass = 0; pass < 16; ++pass) {
        for (std::size_t i = 0; i < z.size(); ++i) z[i] = pseudo_z(i + static_cast<std::size_t>(pass));
        mesh.set_z(z);
        mesh.update_normals();
        CHECK(static_cast<const void*>(mesh.vertices().data()) == v0);
        CHECK(static_cast<const void*>(mesh.indices().data()) == i0);
        CHECK(mesh.vertices().size() == vn0);
        CHECK(mesh.indices().size() == in0);
    }
    // set_axes も再確保しない。
    mesh.set_axes(linspace(-5.f, 5.f, kNx), linspace(-5.f, 5.f, kNy));
    CHECK(static_cast<const void*>(mesh.vertices().data()) == v0);
    CHECK(mesh.vertices().size() == vn0);
}

namespace {

/// 格子全点の法線を解析勾配 (fx, fy) と突き合わせる。
template <class F, class Fx, class Fy>
void check_normals_against_gradient(const std::vector<float>& xs, const std::vector<float>& ys, F f,
                                    Fx fx, Fy fy, double tol) {
    const std::size_t nx = xs.size(), ny = ys.size();
    SurfaceMesh       mesh(nx, ny);
    mesh.set_axes(xs, ys);

    std::vector<float> z(nx * ny);
    for (std::size_t iy = 0; iy < ny; ++iy)
        for (std::size_t ix = 0; ix < nx; ++ix)
            z[iy * nx + ix] =
                static_cast<float>(f(static_cast<double>(xs[ix]), static_cast<double>(ys[iy])));
    mesh.set_z(z);
    mesh.update_normals();

    const auto verts = mesh.vertices();
    REQUIRE(verts.size() == nx * ny);
    for (std::size_t iy = 0; iy < ny; ++iy) {
        for (std::size_t ix = 0; ix < nx; ++ix) {
            const double x = static_cast<double>(xs[ix]);
            const double y = static_cast<double>(ys[iy]);
            const double gx = fx(x, y), gy = fy(x, y);
            const double inv = 1.0 / std::sqrt(gx * gx + gy * gy + 1.0);
            const auto&  v   = verts[iy * nx + ix];
            INFO("(" << ix << "," << iy << ") x=" << x << " y=" << y);
            CHECK_THAT(static_cast<double>(v.nx), WithinAbs(-gx * inv, tol));
            CHECK_THAT(static_cast<double>(v.ny), WithinAbs(-gy * inv, tol));
            CHECK_THAT(static_cast<double>(v.nz), WithinAbs(inv, tol));
        }
    }
}

/// [lo, hi] を power で偏らせた軸（power = 1 で等間隔、> 1 で lo 側が細かい）。
std::vector<float> graded(float lo, float hi, std::size_t n, double power) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(n - 1);
        v[i] = static_cast<float>(static_cast<double>(lo) +
                                  (static_cast<double>(hi) - static_cast<double>(lo)) *
                                      std::pow(u, power));
    }
    return v;
}

}  // namespace

TEST_CASE("SURF-06: normals of z = f(x,y) match the analytic gradient (1e-3)", "[surf][numeric]") {
    SECTION("smooth transcendental surface on a uniform axis") {
        // f(x,y) = 0.4 sin(1.3x) cos(0.9y) + 0.2xy — 境界で曲率が消えない一般の面。
        // 129 点 / [-1.5, 1.5] ⇒ h ≈ 0.0234、打ち切り誤差は h²/6·|f'''| ≈ 1e-4 程度。
        constexpr std::size_t kN = 129;
        check_normals_against_gradient(
            linspace(-1.5f, 1.5f, kN), linspace(-1.5f, 1.5f, kN),
            [](double x, double y) { return 0.4 * std::sin(1.3 * x) * std::cos(0.9 * y) + 0.2 * x * y; },
            [](double x, double y) { return 0.52 * std::cos(1.3 * x) * std::cos(0.9 * y) + 0.2 * y; },
            [](double x, double y) { return -0.36 * std::sin(1.3 * x) * std::sin(0.9 * y) + 0.2 * x; },
            1e-3);
    }

    SECTION("quadratic surface on graded (non-uniform) axes") {
        // 2 次式は 3 点公式が厳密に再現するので、残るのは float の丸めだけ。等間隔でない軸で
        // 素朴な (z₊−z₋)/(x₊−x₋) を使うと f''·(h₊−h₋)/2 の 1 次誤差（ここでは最大 3 割）が出る。
        // x は lo 側が細かい 1.5 乗、y は逆向きの 1.2 乗と、向きの違う偏りを混ぜる。
        constexpr std::size_t kN = 33;
        check_normals_against_gradient(
            graded(0.f, 4.f, kN, 1.5), graded(-2.f, 1.f, kN, 1.2),
            [](double x, double y) {
                return 0.8 * x * x - 0.5 * y * y + 0.4 * x * y + 0.3 * x - 0.2 * y;
            },
            [](double x, double y) { return 1.6 * x + 0.4 * y + 0.3; },
            [](double x, double y) { return -1.0 * y + 0.4 * x - 0.2; }, 1e-3);
    }
}
