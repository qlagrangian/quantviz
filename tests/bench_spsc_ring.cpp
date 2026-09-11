// BENCH — 既定では実行されない（[!benchmark] タグ）。
//   ./quantviz_tests "[!benchmark]" で実行。
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "quantviz/bridge/spsc_ring.hpp"
#include "quantviz/scenes/streaming_model.hpp"

TEST_CASE("BENCH-01: SpscRing push+pop of a StreamingSnapshot", "[!benchmark][ring]") {
    quantviz::bridge::SpscRing<quantviz::scenes::StreamingSnapshot, 1024> ring;
    quantviz::scenes::StreamingSnapshot                                    s{}, out{};
    BENCHMARK("push+pop") {
        ring.try_push(s);
        ring.try_pop(out);
        return out.seq;
    };
}

TEST_CASE("BENCH-02: StreamingModel::step + snapshot", "[!benchmark][stream]") {
    quantviz::scenes::StreamingModel m;
    BENCHMARK("step+snapshot") {
        m.step(1.0 / 252.0);
        return m.snapshot().spot;
    };
}
