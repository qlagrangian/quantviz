#pragma once
// viz/gl/camera.hpp — 目標点のまわりを回る軌道カメラ（vizcore: GUI・GL ヘッダ非依存）。
//
// 責務: (yaw, pitch, distance) の球座標で表した視点から、描画に必要なビュー行列・射影行列と、
// ワールド ⇄ NDC（正規化デバイス座標）の往復を提供する。マウス操作は viz 層が「回転量」「拡大率」
// に翻訳してここへ渡すだけで、状態はすべてこの POD 的な構造体が持つ。
//
// 不正な値はこの層で吸収する（パネルがスライダーで直接メンバへ書き込めるようにするため）:
//   - pitch は (−π/2, π/2) の内側にクランプ。視線が up ベクトル (0,1,0) と平行にならないので
//     look_at が退化しない（GL-02）。
//   - distance は [max(1e-3, znear), 1e3]。下限を znear に合わせてあるので、ズームしきっても
//     目標点が近クリップ面の手前に埋まって消えることはない。
//   - fovy・znear・zfar は `projection()` の中で安全な範囲へ落とす（fovy ∈ (1e-3, π−1e-3)、
//     znear > 0、zfar > znear）。NaN は `!(v > lo)` 形で下限へ倒す（M1 の UI 入力方針と同じ）。
//   - 回転量・拡大率の NaN / ∞ / 非正値は黙って捨てる。
//
// yaw = 0, pitch = 0 のとき視点は target + (0, 0, distance)、すなわちカメラは −z を向く。
//
// 精度について: project → unproject の往復誤差は float32 の NDC z の分解能で決まり、
// zfar/znear が大きいほど粗い。既定（0.05 / 100 = 2000:1）で絶対 2e-5 程度、
// 0.5 / 20 = 40:1 なら 1.4e-6 程度（GL-03）。ピッキング精度が要るシーンは znear を上げる。

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>

#include "quantviz/viz/gl/math.hpp"

namespace quantviz::viz::gl {

struct OrbitCamera {
    /// pitch の上限（極に触れない）。
    static constexpr float kPitchLimit  = std::numbers::pi_v<float> / 2.f - 1e-3f;
    static constexpr float kMinDistance = 1e-3f;
    static constexpr float kMaxDistance = 1e3f;
    static constexpr float kMinFovy     = 1e-3f;
    static constexpr float kMaxFovy     = std::numbers::pi_v<float> - 1e-3f;
    /// znear が 0・負・NaN のときに使う代替値。
    static constexpr float kFallbackZNear = 0.05f;
    /// ワールドの上方向。pitch のクランプによって視線と平行にはならない。
    static constexpr Vec3 kUp{0.f, 1.f, 0.f};

    Vec3  target{0.f, 0.f, 0.f};
    float distance = 3.f;
    float yaw      = 0.f;   ///< rad
    float pitch    = 0.5f;  ///< rad、(−kPitchLimit, kPitchLimit)
    float fovy     = 0.9f;  ///< rad（垂直画角）
    float znear    = 0.05f;
    float zfar     = 100.f;

    /// 目標点を中心に回す。distance は変えない（GL-02）。非有限な入力は無視。
    void rotate(float dyaw, float dpitch) noexcept {
        constexpr float kTwoPi = 2.f * std::numbers::pi_v<float>;
        if (std::isfinite(dyaw)) yaw = std::fmod(yaw + dyaw, kTwoPi);
        if (std::isfinite(dpitch)) pitch = std::clamp(pitch + dpitch, -kPitchLimit, kPitchLimit);
    }

    /// distance *= factor。倍率が正の有限値でなければ何もしない。
    /// 下限は max(kMinDistance, znear): これより寄ると目標点が近クリップ面の手前に出てしまう。
    void zoom(float factor) noexcept {
        if (!(factor > 0.f && factor < std::numeric_limits<float>::infinity())) return;
        distance = std::clamp(distance * factor, std::max(kMinDistance, safe_znear()), kMaxDistance);
    }

    [[nodiscard]] Vec3 eye() const noexcept {
        const float cp = std::cos(pitch), sp = std::sin(pitch);
        const float cy = std::cos(yaw), sy = std::sin(yaw);
        return Vec3{target.x + distance * cp * sy, target.y + distance * sp,
                    target.z + distance * cp * cy};
    }

    [[nodiscard]] Mat4 view() const noexcept { return look_at(eye(), target, kUp); }

    [[nodiscard]] Mat4 projection(float aspect) const noexcept {
        return perspective(safe_fovy(), aspect, safe_znear(), safe_zfar());
    }

    /// ワールド座標 → NDC（x, y, z すべて [−1, 1] が視錐台の内側）。
    [[nodiscard]] Vec3 project(Vec3 world, float aspect) const noexcept {
        return transform_point(projection(aspect) * view(), world);
    }

    /// NDC → ワールド座標。P·V が退化していれば nullopt（GL-03）。
    [[nodiscard]] std::optional<Vec3> unproject(Vec3 ndc, float aspect) const noexcept {
        const std::optional<Mat4> inv = inverse(projection(aspect) * view());
        if (!inv) return std::nullopt;
        return transform_point(*inv, ndc);
    }

    /// 実際に射影へ渡される値（クランプ後）。テストと診断表示のために公開する。
    [[nodiscard]] float safe_fovy() const noexcept { return clamp_low_first(fovy, kMinFovy, kMaxFovy); }
    [[nodiscard]] float safe_znear() const noexcept {
        return (znear > 0.f) ? std::min(znear, kMaxDistance) : kFallbackZNear;
    }
    [[nodiscard]] float safe_zfar() const noexcept {
        const float n = safe_znear();
        return (zfar > n) ? zfar : n * 1000.f;
    }

private:
    /// NaN を下限へ倒す clamp。`std::clamp` は NaN をそのまま返してしまうので使わない。
    static constexpr float clamp_low_first(float v, float lo, float hi) noexcept {
        if (!(v > lo)) return lo;
        return (v < hi) ? v : hi;
    }
};

}  // namespace quantviz::viz::gl
