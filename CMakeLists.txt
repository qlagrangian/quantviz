cmake_minimum_required(VERSION 3.25)
project(quantviz VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(QUANTVIZ_BUILD_VIZ            "Build the ImGui/ImPlot viewer (needs glfw3, imgui, implot)" ON)
option(QUANTVIZ_BUILD_TESTS          "Build unit tests (Catch2)"                                  ON)
option(QUANTVIZ_WARNINGS_AS_ERRORS   "Treat compiler warnings as errors"                          OFF)

# ---------------------------------------------------------------------------
# Warning profile shared by every target we own (third-party code is excluded)
# ---------------------------------------------------------------------------
add_library(quantviz_warnings INTERFACE)
add_library(quantviz::warnings ALIAS quantviz_warnings)
target_compile_options(quantviz_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /Zc:__cplusplus>)
if(QUANTVIZ_WARNINGS_AS_ERRORS)
  target_compile_options(quantviz_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>
    $<$<CXX_COMPILER_ID:MSVC>:/WX>)
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Layer 1: core  — pure numerics, header-only, ZERO external dependencies
# ---------------------------------------------------------------------------
add_library(quantviz_core INTERFACE)
add_library(quantviz::core ALIAS quantviz_core)
target_include_directories(quantviz_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core/include)
target_compile_features(quantviz_core INTERFACE cxx_std_20)

# ---------------------------------------------------------------------------
# Layer 2: bridge — snapshot ring, command queue, sim clock, runner (std only)
# ---------------------------------------------------------------------------
add_library(quantviz_bridge INTERFACE)
add_library(quantviz::bridge ALIAS quantviz_bridge)
target_include_directories(quantviz_bridge INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/bridge/include)
target_link_libraries(quantviz_bridge INTERFACE Threads::Threads)
target_compile_features(quantviz_bridge INTERFACE cxx_std_20)

# ---------------------------------------------------------------------------
# Layer 3: scenes — a Model = core numerics wired to a Snapshot/Command contract
# ---------------------------------------------------------------------------
add_library(quantviz_scenes INTERFACE)
add_library(quantviz::scenes ALIAS quantviz_scenes)
target_include_directories(quantviz_scenes INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/scenes/include)
target_link_libraries(quantviz_scenes INTERFACE quantviz::core quantviz::bridge)

# ---------------------------------------------------------------------------
# Layer 4a: vizcore — pure, GUI-free helpers used by the viewer (History ...)
#           Kept separate so they stay unit-testable without ImGui.
# ---------------------------------------------------------------------------
add_library(quantviz_vizcore INTERFACE)
add_library(quantviz::vizcore ALIAS quantviz_vizcore)
target_include_directories(quantviz_vizcore INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/viz/include)
target_link_libraries(quantviz_vizcore INTERFACE quantviz::scenes)

# ---------------------------------------------------------------------------
# Layer 4b: viz — the only layer allowed to touch ImGui / ImPlot / OpenGL
# ---------------------------------------------------------------------------
if(QUANTVIZ_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(QUANTVIZ_BUILD_VIZ)
  add_subdirectory(viz)
endif()
