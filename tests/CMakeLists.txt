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
  bench_spsc_ring.cpp)

target_link_libraries(quantviz_tests PRIVATE
  quantviz::core quantviz::bridge quantviz::scenes quantviz::vizcore quantviz::warnings
  Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(quantviz_tests)
