// GL-xx — viz/gl/math.hpp・viz/gl/camera.hpp（vizcore の 3D 数学。GUI 非依存）の仕様テスト
//
// 列優先 4×4（OpenGL 規約、m[col*4 + row]）の look-at / perspective を手計算の参照値と突き合わせ、
// 軌道カメラの回転が目標点との距離を保つこと、project → unproject が往復すること、
// 射影の [0][0] がアスペクト比の逆数に比例することを確かめる。すべて float（GL に上げる型）。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>

#include "quantviz/viz/gl/camera.hpp"
#include "quantviz/viz/gl/math.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::viz::gl::inverse;
using quantviz::viz::gl::length;
using quantviz::viz::gl::look_at;
using quantviz::viz::gl::Mat4;
using quantviz::viz::gl::OrbitCamera;
using quantviz::viz::gl::perspective;
using quantviz::viz::gl::transform_point;
using quantviz::viz::gl::Vec3;

namespace {

constexpr float kPi = std::numbers::pi_v<float>;

/// 0 を含む参照値を「相対 1e-6、ただし 0 近傍は絶対 1e-6」で比較する（§2.4 の数表値扱い）。
void check_matrix(const Mat4& got, const std::array<double, 16>& want, double tol = 1e-6) {
    for (std::size_t i = 0; i < 16; ++i) {
        INFO("element " << i);
        CHECK_THAT(static_cast<double>(got.m[i]), WithinRel(want[i], tol) || WithinAbs(want[i], tol));
    }
}

}  // namespace

TEST_CASE("GL-01: look_at / perspective match hand-computed reference matrices (rel 1e-6)",
          "[gl][numeric]") {
    SECTION("look_at from (0,0,5) to the origin with +y up is a pure translation of -5 along z") {
        // f = (0,0,-1), s = f×up = (1,0,0), u = s×f = (0,1,0);
        // 平行移動は (-s·eye, -u·eye, f·eye) = (0, 0, -5)。
        const Mat4 v = look_at(Vec3{0.f, 0.f, 5.f}, Vec3{0.f, 0.f, 0.f}, Vec3{0.f, 1.f, 0.f});
        check_matrix(v, {1, 0, 0, 0,  //
                         0, 1, 0, 0,  //
                         0, 0, 1, 0,  //
                         0, 0, -5, 1});
        // カメラはビュー空間の原点、注視点は −z 側に distance だけ離れる（右手系・−z を向く）。
        const Vec3 eye_v = transform_point(v, Vec3{0.f, 0.f, 5.f});
        const Vec3 tgt_v = transform_point(v, Vec3{0.f, 0.f, 0.f});
        CHECK_THAT(static_cast<double>(eye_v.z), WithinAbs(0.0, 1e-6));
        CHECK_THAT(static_cast<double>(tgt_v.z), WithinAbs(-5.0, 1e-6));
    }

    SECTION("look_at from (3,4,0) to the origin with +z up") {
        // f = (-0.6,-0.8,0), s = f×up = (-0.8,0.6,0), u = s×f = (0,0,1), f·eye = -5。
        const Mat4 v = look_at(Vec3{3.f, 4.f, 0.f}, Vec3{0.f, 0.f, 0.f}, Vec3{0.f, 0.f, 1.f});
        check_matrix(v, {-0.8, 0.0, 0.6, 0.0,  //
                         0.6, 0.0, 0.8, 0.0,   //
                         0.0, 1.0, 0.0, 0.0,   //
                         0.0, 0.0, -5.0, 1.0});
        const Vec3 eye_v = transform_point(v, Vec3{3.f, 4.f, 0.f});
        CHECK_THAT(static_cast<double>(eye_v.x), WithinAbs(0.0, 1e-6));
        CHECK_THAT(static_cast<double>(eye_v.y), WithinAbs(0.0, 1e-6));
        CHECK_THAT(static_cast<double>(eye_v.z), WithinAbs(0.0, 1e-6));
    }

    SECTION("perspective(pi/2, 1, 1, 10)") {
        // f = 1/tan(pi/4) = 1; [2][2] = (zf+zn)/(zn-zf) = -11/9; [2][3] = 2·zf·zn/(zn-zf) = -20/9。
        const Mat4 p = perspective(kPi / 2.f, 1.f, 1.f, 10.f);
        check_matrix(p, {1, 0, 0, 0,             //
                         0, 1, 0, 0,             //
                         0, 0, -11.0 / 9.0, -1,  //
                         0, 0, -20.0 / 9.0, 0});
        // 近クリップ面 z=-1 → NDC z=-1、遠クリップ面 z=-10 → NDC z=+1。
        CHECK_THAT(static_cast<double>(transform_point(p, Vec3{0.f, 0.f, -1.f}).z),
                   WithinAbs(-1.0, 1e-6));
        CHECK_THAT(static_cast<double>(transform_point(p, Vec3{0.f, 0.f, -10.f}).z),
                   WithinAbs(1.0, 1e-6));
    }

    SECTION("perspective(pi/3, 16/9, 0.5, 50) — [0][0] = f/aspect, [1][1] = f") {
        const float aspect = 16.f / 9.f;
        const Mat4  p      = perspective(kPi / 3.f, aspect, 0.5f, 50.f);
        const double f     = 1.0 / std::tan(std::numbers::pi / 6.0);  // = sqrt(3)
        CHECK_THAT(static_cast<double>(p.at(0, 0)), WithinRel(f / (16.0 / 9.0), 1e-6));
        CHECK_THAT(static_cast<double>(p.at(1, 1)), WithinRel(f, 1e-6));
        CHECK_THAT(static_cast<double>(p.at(2, 2)), WithinRel(50.5 / -49.5, 1e-6));
        CHECK_THAT(static_cast<double>(p.at(2, 3)), WithinRel(50.0 / -49.5, 1e-6));
        CHECK_THAT(static_cast<double>(p.at(3, 2)), WithinRel(-1.0, 1e-6));
        CHECK_THAT(static_cast<double>(p.at(3, 3)), WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("GL-02: orbit rotation keeps the eye at a constant distance from the target",
          "[gl][property]") {
    OrbitCamera cam;
    cam.target   = Vec3{1.f, -2.f, 0.5f};
    cam.distance = 4.25f;
    cam.yaw      = 0.3f;
    cam.pitch    = -0.2f;

    const float d0 = cam.distance;

    SECTION("any sequence of rotations preserves |eye - target| and the distance field") {
        const float deltas[] = {0.1f, -0.7f, 2.5f, -3.3f, 0.0f, 12.0f, -40.0f};
        for (const float dy : deltas) {
            for (const float dp : deltas) {
                cam.rotate(dy, dp);
                const Vec3 r = Vec3{cam.eye().x - cam.target.x, cam.eye().y - cam.target.y,
                                    cam.eye().z - cam.target.z};
                INFO("dyaw " << dy << " dpitch " << dp << " pitch " << cam.pitch);
                CHECK_THAT(static_cast<double>(cam.distance), WithinRel(static_cast<double>(d0), 1e-6));
                CHECK_THAT(static_cast<double>(length(r)), WithinRel(static_cast<double>(d0), 1e-5));
                // pitch は極（±π/2）に触れない: up ベクトルと視線が縮退しない。
                CHECK(std::abs(cam.pitch) < kPi / 2.f);
            }
        }
    }

    SECTION("the view matrix stays finite and puts the target at -distance on the view z axis") {
        cam.rotate(0.9f, 100.f);  // pitch をクランプにぶつける
        const Mat4 v = cam.view();
        for (const float e : v.m) CHECK(std::isfinite(e));
        const Vec3 t = transform_point(v, cam.target);
        CHECK_THAT(static_cast<double>(t.x), WithinAbs(0.0, 1e-4));
        CHECK_THAT(static_cast<double>(t.y), WithinAbs(0.0, 1e-4));
        CHECK_THAT(static_cast<double>(t.z), WithinRel(-static_cast<double>(d0), 1e-5));
    }

    SECTION("zoom scales the distance and clamps it to [max(1e-3, znear), 1e3]") {
        cam.zoom(2.f);
        CHECK_THAT(static_cast<double>(cam.distance), WithinRel(2.0 * static_cast<double>(d0), 1e-6));
        // 寄りきっても目標点は近クリップ面の外側 = まだ見える（既定 znear 0.05 は kMinDistance より大きい）。
        for (int i = 0; i < 100; ++i) cam.zoom(0.5f);
        CHECK(cam.distance >= cam.znear);
        CHECK(cam.distance >= OrbitCamera::kMinDistance);
        // znear が kMinDistance より小さいカメラでは kMinDistance が効く。
        OrbitCamera near_cam;
        near_cam.znear = 1e-5f;
        for (int i = 0; i < 100; ++i) near_cam.zoom(0.5f);
        CHECK(near_cam.distance >= OrbitCamera::kMinDistance);
        for (int i = 0; i < 200; ++i) cam.zoom(2.f);
        CHECK(cam.distance <= 1e3f);
        // 不正な倍率（0・負・NaN）は無視する。
        const float before = cam.distance;
        cam.zoom(0.f);
        cam.zoom(-1.f);
        cam.zoom(std::numeric_limits<float>::quiet_NaN());
        CHECK(cam.distance == before);
    }
}

TEST_CASE("GL-03: unproject(project(p)) recovers p (abs 1e-5)", "[gl][numeric]") {
    // znear / zfar は既定（0.05 / 100）より締めてある。float の NDC z は近遠比が大きいほど
    // 粗くなり、既定値だと往復誤差が 1.4e-5 と許容 1e-5 を超える（浮動小数の分解能そのもので、
    // 実装の誤りではない）。この設定（0.5 / 20）での実測最悪誤差は 1.4e-6 ＝ 許容の 1/7。
    OrbitCamera cam;
    cam.target   = Vec3{0.f, 0.f, 0.f};
    cam.distance = 3.f;
    cam.yaw      = 0.7f;
    cam.pitch    = 0.4f;
    cam.fovy     = 0.9f;
    cam.znear    = 0.5f;
    cam.zfar     = 20.f;

    const float aspect = 16.f / 9.f;

    SECTION("round trip over a grid of world points inside the frustum") {
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                for (int k = -2; k <= 2; ++k) {
                    const Vec3 p{static_cast<float>(i) * 0.5f, static_cast<float>(j) * 0.5f,
                                 static_cast<float>(k) * 0.5f};
                    const Vec3                ndc  = cam.project(p, aspect);
                    const std::optional<Vec3> back = cam.unproject(ndc, aspect);
                    REQUIRE(back.has_value());
                    INFO("p = (" << p.x << "," << p.y << "," << p.z << ")");
                    CHECK_THAT(static_cast<double>(back->x), WithinAbs(static_cast<double>(p.x), 1e-5));
                    CHECK_THAT(static_cast<double>(back->y), WithinAbs(static_cast<double>(p.y), 1e-5));
                    CHECK_THAT(static_cast<double>(back->z), WithinAbs(static_cast<double>(p.z), 1e-5));
                }
            }
        }
    }

    SECTION("round trip at the shipping defaults (znear 0.05 / zfar 100) — abs 5e-5") {
        // 出荷時の既定は zfar/znear = 2000:1。float32 の NDC z の分解能がここで効き、往復誤差は
        // 上の 40:1 の設定（1.4e-6）より一桁以上大きい 2.1e-5 程度になる。実装の誤りではなく
        // 深度の条件数そのものなので、既定値での保証はこの緩い境界（5e-5）で宣言しておく。
        OrbitCamera d;  // distance 3, yaw 0, pitch 0.5, fovy 0.9, znear 0.05, zfar 100
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                for (int k = -2; k <= 2; ++k) {
                    const Vec3 p{static_cast<float>(i) * 0.5f, static_cast<float>(j) * 0.5f,
                                 static_cast<float>(k) * 0.5f};
                    const std::optional<Vec3> back = d.unproject(d.project(p, aspect), aspect);
                    REQUIRE(back.has_value());
                    INFO("p = (" << p.x << "," << p.y << "," << p.z << ")");
                    CHECK_THAT(static_cast<double>(back->x), WithinAbs(static_cast<double>(p.x), 5e-5));
                    CHECK_THAT(static_cast<double>(back->y), WithinAbs(static_cast<double>(p.y), 5e-5));
                    CHECK_THAT(static_cast<double>(back->z), WithinAbs(static_cast<double>(p.z), 5e-5));
                }
            }
        }
    }

    SECTION("inverse(M) * M is the identity") {
        const Mat4                p  = cam.projection(aspect);
        const Mat4                v  = cam.view();
        const Mat4                m  = p * v;
        const std::optional<Mat4> mi = inverse(m);
        REQUIRE(mi.has_value());
        const Mat4 id = *mi * m;
        for (std::size_t r = 0; r < 4; ++r) {
            for (std::size_t c = 0; c < 4; ++c) {
                const double want = (r == c) ? 1.0 : 0.0;
                INFO("(" << r << "," << c << ")");
                CHECK_THAT(static_cast<double>(id.at(r, c)), WithinAbs(want, 1e-5));
            }
        }
    }

    SECTION("a singular matrix has no inverse") {
        Mat4 z{};  // 全要素 0
        CHECK_FALSE(inverse(z).has_value());
        Mat4 rank3     = Mat4::identity();
        rank3.at(2, 2) = 0.f;
        CHECK_FALSE(inverse(rank3).has_value());
    }

    SECTION("the target projects to the centre of the screen") {
        const Vec3 c = cam.project(cam.target, aspect);
        CHECK_THAT(static_cast<double>(c.x), WithinAbs(0.0, 1e-5));
        CHECK_THAT(static_cast<double>(c.y), WithinAbs(0.0, 1e-5));
    }
}

TEST_CASE("GL-04: the projection's (0,0) entry scales as 1 / aspect", "[gl][unit]") {
    const float fovy = 0.9f;
    const double f   = 1.0 / std::tan(static_cast<double>(fovy) / 2.0);

    for (const float aspect : {0.5f, 1.0f, 16.f / 9.f, 3.0f, 10.0f}) {
        const Mat4 p = perspective(fovy, aspect, 0.1f, 100.f);
        INFO("aspect " << aspect);
        // [0][0]·aspect はアスペクト比によらず 1/tan(fovy/2) で一定 ⇒ [0][0] ∝ 1/aspect。
        CHECK_THAT(static_cast<double>(p.at(0, 0)) * static_cast<double>(aspect), WithinRel(f, 1e-6));
        // 縦方向 [1][1] はアスペクト比に依存しない。
        CHECK_THAT(static_cast<double>(p.at(1, 1)), WithinRel(f, 1e-6));
    }

    SECTION("OrbitCamera::projection uses the same convention") {
        OrbitCamera cam;
        cam.fovy = fovy;
        for (const float aspect : {0.75f, 2.0f}) {
            const Mat4 p = cam.projection(aspect);
            INFO("aspect " << aspect);
            CHECK_THAT(static_cast<double>(p.at(0, 0)) * static_cast<double>(aspect), WithinRel(f, 1e-6));
        }
    }
}
