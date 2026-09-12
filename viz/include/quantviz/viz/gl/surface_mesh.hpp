#pragma once
// viz/gl/surface_mesh.hpp — n_x × n_y の高さ格子 z = f(x, y) を、GL にそのまま上げられる
// 頂点バッファとインデックスバッファに変換する（vizcore: GUI・GL ヘッダ非依存）。
//
// 責務: 軸（x, y）は固定、z だけが毎フレーム入れ替わる、という使い方に最適化する。確保は
// コンストラクタの一度きりで、`set_z` / `update_normals` は既存のバッファを書き換えるだけ
// （SURF-05）。したがって描画側は VBO/EBO を作り直さず `glBufferSubData` で上書きできる。
//
// 走査順は y メジャー行: 頂点・z 配列ともに index = iy * n_x + ix（ix が連続）。
// 頂点は位置と法線をインターリーブした 6 float = 24 B で、そのまま stride 24 の VBO になる。
// 三角形は 2(n_x−1)(n_y−1) 枚。xs・ys が昇順なら +z から見て反時計回り（CCW = front face）。
//
// 法線は ∂z/∂x・∂z/∂y の差分から (−z_x, −z_y, 1) を正規化して得る。内点も端点も「隣接 3 点を
// 通る 2 次式のその点における傾き」を使う: 内点は差分商の間隔加重平均、端点は片側 3 点。
// 等間隔ならそれぞれ (z₊−z₋)/2h と (−3z₀+4z₁−z₂)/2h に一致し、どちらも 2 次精度。軸が不等間隔でも
// 2 次精度を保つ（素朴な (z₊−z₋)/(x₊−x₋) は中点の傾きなので、不等間隔だと 1 次精度に落ちる）。

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace quantviz::viz::gl {

class SurfaceMesh {
public:
    /// GL の頂点属性そのまま（location 0 = 位置、location 1 = 法線、stride 24 B）。
    struct Vertex {
        float x, y, z, nx, ny, nz;
    };
    static_assert(sizeof(Vertex) == 24, "Vertex must stay a tightly packed 6-float VBO record");
    static_assert(alignof(Vertex) == 4, "Vertex must not gain padding");

    /// n_x, n_y はいずれも 2 以上（1 点では面にならない）。小さい値は 2 に切り上げる
    /// （Debug では assert で気づける。Release でも UB にはしない）。
    SurfaceMesh(std::size_t n_x, std::size_t n_y)
        : n_x_(n_x < 2 ? std::size_t{2} : n_x),
          n_y_(n_y < 2 ? std::size_t{2} : n_y),
          xs_(n_x_),
          ys_(n_y_),
          vertices_(n_x_ * n_y_),
          indices_(6 * (n_x_ - 1) * (n_y_ - 1)) {
        assert(n_x >= 2 && n_y >= 2 && "SurfaceMesh needs at least a 2x2 grid");
        assert(n_x_ * n_y_ <= 0xFFFFFFFFu && "vertex count must fit in a 32-bit index");

        // 既定の軸は 0, 1, 2, ...（縮退しない間隔。set_axes を呼ぶまでの安全な初期値）。
        for (std::size_t ix = 0; ix < n_x_; ++ix) xs_[ix] = static_cast<float>(ix);
        for (std::size_t iy = 0; iy < n_y_; ++iy) ys_[iy] = static_cast<float>(iy);
        refresh_positions();
        for (Vertex& v : vertices_) {
            v.nx = 0.f;
            v.ny = 0.f;
            v.nz = 1.f;
        }
        build_indices();
    }

    /// 軸を差し替える。xs は n_x 個、ys は n_y 個。サイズが合わなければ何もしない。
    /// 前提: xs・ys は有限かつ狭義単調増加（昇順でないとインデックスの CCW が崩れる）。
    void set_axes(std::span<const float> xs, std::span<const float> ys) noexcept {
        assert(xs.size() == n_x_ && ys.size() == n_y_);
        if (xs.size() != n_x_ || ys.size() != n_y_) return;
        std::copy(xs.begin(), xs.end(), xs_.begin());
        std::copy(ys.begin(), ys.end(), ys_.begin());
        refresh_positions();
    }

    /// 高さを差し替える（row-major、index = iy * n_x + ix）。サイズが合わなければ何もしない。
    /// 再確保はしない（SURF-05）。法線は古いままなので `update_normals()` を続けて呼ぶ。
    ///
    /// 前提: z はすべて有限。NaN / ∞ を入れると法線が NaN になって面が黒く落ちるが、
    /// 毎フレームの経路で全要素を走査したくないので Release では検査しない（Debug の assert のみ）。
    /// モデル側（FDM・ボラ面）が発散を検知して status に出すのが正しい層分け。
    void set_z(std::span<const float> z_row_major) noexcept {
        assert(z_row_major.size() == vertices_.size());
        if (z_row_major.size() != vertices_.size()) return;
        for (std::size_t i = 0; i < vertices_.size(); ++i) {
            assert(std::isfinite(z_row_major[i]) && "set_z requires finite heights");
            vertices_[i].z = z_row_major[i];
        }
    }

    /// 現在の z から法線を計算し直す。ゼロアロケーション。
    void update_normals() noexcept {
        for (std::size_t iy = 0; iy < n_y_; ++iy) {
            for (std::size_t ix = 0; ix < n_x_; ++ix) {
                const float dzdx = slope_x(ix, iy);
                const float dzdy = slope_y(ix, iy);
                // (−z_x, −z_y, 1) の長さは必ず 1 以上なので 0 除算にならない。
                const float inv = 1.f / std::sqrt(dzdx * dzdx + dzdy * dzdy + 1.f);
                Vertex&     v   = vertices_[iy * n_x_ + ix];
                v.nx            = -dzdx * inv;
                v.ny            = -dzdy * inv;
                v.nz            = inv;
            }
        }
    }

    [[nodiscard]] std::span<const Vertex> vertices() const noexcept { return vertices_; }
    [[nodiscard]] std::span<const std::uint32_t> indices() const noexcept { return indices_; }
    [[nodiscard]] std::size_t n_x() const noexcept { return n_x_; }
    [[nodiscard]] std::size_t n_y() const noexcept { return n_y_; }
    [[nodiscard]] std::size_t triangle_count() const noexcept { return indices_.size() / 3; }

private:
    void refresh_positions() noexcept {
        for (std::size_t iy = 0; iy < n_y_; ++iy)
            for (std::size_t ix = 0; ix < n_x_; ++ix) {
                Vertex& v = vertices_[iy * n_x_ + ix];
                v.x       = xs_[ix];
                v.y       = ys_[iy];
            }
    }

    void build_indices() noexcept {
        std::size_t k = 0;
        for (std::size_t iy = 0; iy + 1 < n_y_; ++iy)
            for (std::size_t ix = 0; ix + 1 < n_x_; ++ix) {
                const std::uint32_t v00 = static_cast<std::uint32_t>(iy * n_x_ + ix);
                const std::uint32_t v10 = v00 + 1u;
                const std::uint32_t v01 = static_cast<std::uint32_t>(v00 + n_x_);
                const std::uint32_t v11 = v01 + 1u;
                // xs・ys が昇順なら両方とも +z から見て CCW。
                indices_[k++] = v00;
                indices_[k++] = v10;
                indices_[k++] = v11;
                indices_[k++] = v00;
                indices_[k++] = v11;
                indices_[k++] = v01;
            }
    }

    [[nodiscard]] float z_at(std::size_t ix, std::size_t iy) const noexcept {
        return vertices_[iy * n_x_ + ix].z;
    }

    /// 2 点の差分商。座標差が 0 なら 0（縮退した軸で NaN を出さない）。
    [[nodiscard]] static float divided_difference(float x0, float z0, float x1, float z1) noexcept {
        const float d = x1 - x0;
        return d != 0.f ? (z1 - z0) / d : 0.f;
    }

    /// (xm,zm), (x0,z0), (xp,zp) を通る 2 次式の x0 における傾き（内点用）。
    /// 両側の差分商を間隔で加重平均する: p'(x0) = (h2·d1 + h1·d2)/(h1+h2)。
    /// 等間隔なら (zp−zm)/(xp−xm) と一致し、不等間隔でも 2 次精度を保つ（素朴な形は
    /// 区間の中点での傾きなので、h1 ≠ h2 だと f''·(h2−h1)/2 の誤差が残る）。
    /// 差分商の形なので z が一定なら d1 = d2 = 0 で厳密に 0（SURF-04）。
    [[nodiscard]] static float central_slope(float xm, float zm, float x0, float z0, float xp,
                                             float zp) noexcept {
        const float h1 = x0 - xm;
        const float h2 = xp - x0;
        const float hs = h1 + h2;
        if (h1 == 0.f || h2 == 0.f || hs == 0.f) return divided_difference(xm, zm, xp, zp);
        const float d1 = (z0 - zm) / h1;  // 差分商 [xm, x0]
        const float d2 = (zp - z0) / h2;  // 差分商 [x0, xp]
        return (h2 * d1 + h1 * d2) / hs;
    }

    /// (x0,z0), (x1,z1), (x2,z2) を通る 2 次式の x0 における傾き（端点用の片側 3 点）。
    /// Newton の差分商で書く: p'(x0) = d1 − h1·(d12 − d1)/(h1+h2)。z の 1 次結合として展開した
    /// (−3z₀+4z₁−z₂)/2h と数学的には同じだが、差分の形なら z が一定のとき厳密に 0 になる
    /// （展開形は大きな係数どうしの打ち消しで 1e-7 程度の残差が出る: SURF-04）。
    [[nodiscard]] static float one_sided_slope(float x0, float z0, float x1, float z1, float x2,
                                               float z2) noexcept {
        const float h1 = x1 - x0;
        const float h2 = x2 - x1;
        const float hs = h1 + h2;
        if (h1 == 0.f || h2 == 0.f || hs == 0.f) return divided_difference(x0, z0, x1, z1);
        const float d1  = (z1 - z0) / h1;  // 差分商 [x0, x1]
        const float d12 = (z2 - z1) / h2;  // 差分商 [x1, x2]
        const float w   = h1 / hs;
        return d1 * (1.f + w) - d12 * w;
    }

    [[nodiscard]] float slope_x(std::size_t ix, std::size_t iy) const noexcept {
        if (n_x_ < 3) return divided_difference(xs_[0], z_at(0, iy), xs_[1], z_at(1, iy));
        if (ix == 0)
            return one_sided_slope(xs_[0], z_at(0, iy), xs_[1], z_at(1, iy), xs_[2], z_at(2, iy));
        if (ix + 1 == n_x_)
            return one_sided_slope(xs_[ix], z_at(ix, iy), xs_[ix - 1], z_at(ix - 1, iy), xs_[ix - 2],
                                   z_at(ix - 2, iy));
        return central_slope(xs_[ix - 1], z_at(ix - 1, iy), xs_[ix], z_at(ix, iy), xs_[ix + 1],
                             z_at(ix + 1, iy));
    }

    [[nodiscard]] float slope_y(std::size_t ix, std::size_t iy) const noexcept {
        if (n_y_ < 3) return divided_difference(ys_[0], z_at(ix, 0), ys_[1], z_at(ix, 1));
        if (iy == 0)
            return one_sided_slope(ys_[0], z_at(ix, 0), ys_[1], z_at(ix, 1), ys_[2], z_at(ix, 2));
        if (iy + 1 == n_y_)
            return one_sided_slope(ys_[iy], z_at(ix, iy), ys_[iy - 1], z_at(ix, iy - 1), ys_[iy - 2],
                                   z_at(ix, iy - 2));
        return central_slope(ys_[iy - 1], z_at(ix, iy - 1), ys_[iy], z_at(ix, iy), ys_[iy + 1],
                             z_at(ix, iy + 1));
    }

    std::size_t                n_x_;
    std::size_t                n_y_;
    std::vector<float>         xs_;
    std::vector<float>         ys_;
    std::vector<Vertex>        vertices_;
    std::vector<std::uint32_t> indices_;
};

}  // namespace quantviz::viz::gl
