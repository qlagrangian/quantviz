# Catch2 v3: vcpkg があれば find_package、無ければ FetchContent で取得する
find_package(Catch2 3 CONFIG QUIET)
if(NOT Catch2_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()

add_executable(quantviz_tests
  test_spsc_ring.cpp
  test_sim_clock.cpp
  test_runner.cpp
  test_gbm.cpp
  test_welford.cpp
  test_ewma.cpp
  test_streaming_model.cpp
  test_history.cpp
  bench_spsc_ring.cpp
  # ---- M1
  test_viz.cpp
  test_black_scholes.cpp
  bench_black_scholes.cpp
  test_optim.cpp
  test_garch.cpp
  test_kalman.cpp
  test_greeks.cpp
  # ---- M2
  test_triple_buffer.cpp
  test_gl_math.cpp
  test_surface_mesh.cpp
  bench_surface_mesh.cpp
  test_tridiag.cpp
  test_fdm_cn.cpp
  test_vol_surface.cpp
  test_fdm_scene.cpp
  # ---- M3
  test_order_book.cpp
  bench_order_book.cpp
  test_hawkes.cpp
  test_history2d.cpp
  test_lob_scene.cpp
  # ---- M4
  test_lsm.cpp
  test_almgren_chriss.cpp
  test_hjb_merton.cpp
  test_lsm_scene.cpp
  test_exec_scene.cpp
  test_hjb_scene.cpp
  # ---- M5
  test_perf.cpp)

target_link_libraries(quantviz_tests PRIVATE
  quantviz::core quantviz::bridge quantviz::scenes quantviz::vizcore quantviz::warnings
  Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(quantviz_tests)
