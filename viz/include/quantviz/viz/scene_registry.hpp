#pragma once
// viz/scene_registry.hpp — シーン（Runner + Panel の対）の型消去と切替（vizcore: GUI 非依存）
//
// 責務: 「どのシーンを走らせるか」を 1 か所に集める。`Scene` は Runner と Panel の組を型消去した
// インタフェース、`RunnerScene<M, Panel>` はその唯一の具象実装（start/stop/running を Runner に、
// draw を Panel に委譲する）、`SceneRegistry` は名前 → 生成関数の登録簿で、選択時に
// **前のシーンを stop() してから破棄し、新しいシーンを start() する**（VIZ-01/02）。
// ImGui に触れないので単体テストできる。ウィンドウ／ウィジェットは viz 層の責務。

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/bridge/runner.hpp"

namespace quantviz::viz {

/// Runner + Panel の対を型消去したシーン。フレームループはこれしか知らない。
class Scene {
public:
    Scene()                        = default;
    virtual ~Scene()               = default;
    Scene(const Scene&)            = delete;
    Scene& operator=(const Scene&) = delete;

    virtual void start()         = 0;  ///< 計算スレッドを開始する（runner.start()）
    virtual void stop()          = 0;  ///< 計算スレッドを停止して join する（runner.stop()）
    virtual bool running() const = 0;
    virtual void draw()          = 0;  ///< 毎フレーム 1 回: panel.draw(runner)
};

/// 具象シーンの共通実装。Panel は `void draw(Runner&)` を持てばよい（Model には触らない）。
///
/// 不変条件: 計算スレッドは panel_ より先に必ず止める。宣言順は runner_ → panel_ なので破棄は
/// panel_ → runner_ の順になる。デストラクタ本体で stop()（= join）してから破棄に入ることで、
/// 「走っているスレッドがある間に描画側の状態が消えている」状態を作らない。
template <bridge::Model M, class Panel, std::size_t SnapCap = 4096, std::size_t CmdCap = 256>
class RunnerScene final : public Scene {
public:
    using RunnerType = bridge::Runner<M, SnapCap, CmdCap>;

    RunnerScene(M model, bridge::RunnerConfig cfg, Panel panel)
        : runner_(std::move(model), cfg), panel_(std::move(panel)) {}

    /// Panel をその場で構築する版（History を抱えた大きな Panel を値でスタックに積まないため）。
    template <class... PanelArgs>
    RunnerScene(M model, bridge::RunnerConfig cfg, std::in_place_t, PanelArgs&&... args)
        : runner_(std::move(model), cfg), panel_(std::forward<PanelArgs>(args)...) {}

    ~RunnerScene() override { runner_.stop(); }  // panel_ の破棄前に join する（上の不変条件）

    void start() override { runner_.start(); }
    void stop() override { runner_.stop(); }
    bool running() const override { return runner_.running(); }
    void draw() override { panel_.draw(runner_); }

private:
    RunnerType runner_;
    Panel      panel_;
};

/// 名前 → シーン生成関数の登録簿。生きているシーンは常に高々 1 つ。
class SceneRegistry {
public:
    using Factory = std::function<std::unique_ptr<Scene>()>;

    /// 未選択を表す index。
    static constexpr std::size_t kNone = static_cast<std::size_t>(-1);

    /// 登録順は列挙順（VIZ-01）。生成はここでは行わない（select() まで遅延）。
    void add(std::string name, Factory make) {
        names_.push_back(std::move(name));
        factories_.push_back(std::move(make));
    }

    const std::vector<std::string>& names() const noexcept { return names_; }
    std::size_t                     size() const noexcept { return names_.size(); }

    /// index のシーンを生成して start() する。直前のシーンは stop() してから破棄する（VIZ-02）。
    /// 同じ index を再選択した場合も作り直す（Reset と同義）。範囲外は無視して false を返す。
    /// 返値: その index のシーンが走り始めたら true。範囲外、または factory が nullptr を返したら false。
    ///
    /// 再入不可: 生成関数の中や `current()->draw()` の中から呼んではならない（描画中に自分を破棄する）。
    /// 呼ぶのはフレームの外側（メニュー処理など）から。
    bool select(std::size_t index) {
        if (index >= factories_.size()) return false;
        Factory make = factories_[index];  // 生成中に add() されても壊れないよう控えを取る
        if (current_) {
            current_->stop();  // 計算スレッドを止めてから
            current_.reset();  // 破棄する（2 つのシーンが同時に走らないように順序が重要）
        }
        index_   = kNone;
        current_ = make ? make() : nullptr;
        if (!current_) return false;  // 生成に失敗した登録は未選択のまま扱う
        index_ = index;
        current_->start();
        return true;
    }

    Scene*       current() noexcept { return current_.get(); }
    const Scene* current() const noexcept { return current_.get(); }
    std::size_t  current_index() const noexcept { return index_; }

private:
    std::vector<std::string> names_;
    std::vector<Factory>     factories_;
    std::unique_ptr<Scene>   current_;
    std::size_t              index_ = kNone;
};

}  // namespace quantviz::viz
