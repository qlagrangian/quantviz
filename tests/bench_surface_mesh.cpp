// BENCH-04 — 既定では実行されない（[!benchmark] タグ）。
//   ./quantviz_tests "[!benchmark][surf]" で実行。
// 200×200（= 40,000 頂点、79,202 三角形）の面を毎フレーム作り直す想定の計測。
// 目標 < 2 ms（60 fps の 1 フレーム 16.7 ms のうち CPU 側の面更新に使ってよい分）。
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "quantviz/viz/gl/surface_mesh.hpp"

using quantviz::viz::gl::SurfaceMesh;

TEST_CASE("BENCH-04: 200x200 surface set_z + update_normals", "[!benchmark][surf]") {
    constexpr std::size_t kN = 200;

    std::vector<float> xs(kN), ys(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        xs[i] = -1.f + 2.f * static_cast<float>(i) / static_cast<float>(kN - 1);
        ys[i] = xs[i];
    }

    SurfaceMesh mesh(kN, kN);
    mesh.set_axes(xs, ys);

    std::vector<float> z(kN * kN);
    for (std::size_t iy = 0; iy < kN; ++iy)
        for (std::size_t ix = 0; ix < kN; ++ix)
            z[iy * kN + ix] = std::sin(3.f * xs[ix]) * std::cos(2.f * ys[iy]);

    BENCHMARK("set_z + update_normals (200x200)") {
        mesh.set_z(z);
        mesh.update_normals();
        return mesh.vertices()[kN * kN - 1].nz;
    };

    BENCHMARK("set_z only (200x200)") {
        mesh.set_z(z);
        return mesh.vertices()[kN * kN - 1].z;
    };

    BENCHMARK("update_normals only (200x200)") {
        mesh.update_normals();
        return mesh.vertices()[kN * kN - 1].nz;
    };
}
