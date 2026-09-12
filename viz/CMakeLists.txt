# quantviz_viz — the viewer. The ONLY target allowed to include ImGui / ImPlot / GLFW / OpenGL.
#
# Dependency resolution order:
#   1. vcpkg (or any CMake package)  : find_package(imgui / implot / glfw3 CONFIG)
#   2. Fallback                       : FetchContent imgui + implot from GitHub and build them here;
#                                       GLFW from the system via pkg-config (apt install libglfw3-dev)

find_package(OpenGL REQUIRED)

# --- GLFW ------------------------------------------------------------------
find_package(glfw3 CONFIG QUIET)
if(glfw3_FOUND)
  set(QUANTVIZ_GLFW_TARGET glfw)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GLFW3 REQUIRED IMPORTED_TARGET glfw3)
  set(QUANTVIZ_GLFW_TARGET PkgConfig::GLFW3)
  message(STATUS "quantviz: glfw3 via pkg-config")
endif()

# --- ImGui / ImPlot --------------------------------------------------------
find_package(imgui  CONFIG QUIET)
find_package(implot CONFIG QUIET)
if(imgui_FOUND AND implot_FOUND)
  set(QUANTVIZ_IMGUI_TARGETS imgui::imgui implot::implot)
  message(STATUS "quantviz: imgui/implot via find_package")
else()
  include(FetchContent)
  FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.9
    GIT_SHALLOW    TRUE)
  FetchContent_Declare(implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG        v0.16
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(imgui implot)

  add_library(quantviz_imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
    ${implot_SOURCE_DIR}/implot_demo.cpp)
  target_include_directories(quantviz_imgui SYSTEM PUBLIC
    ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends ${implot_SOURCE_DIR})
  target_link_libraries(quantviz_imgui PUBLIC ${QUANTVIZ_GLFW_TARGET} OpenGL::GL)
  set_target_properties(quantviz_imgui PROPERTIES CXX_STANDARD 17)   # third-party: no warning profile
  set(QUANTVIZ_IMGUI_TARGETS quantviz_imgui)
  message(STATUS "quantviz: imgui/implot built from source (FetchContent)")
endif()

# --- The viewer ------------------------------------------------------------
add_executable(quantviz_viz
  src/main.cpp
  src/panels/streaming_panel.cpp
  src/panels/kalman_panel.cpp
  src/panels/garch_panel.cpp
  src/panels/greeks_panel.cpp
  src/panels/vol_surface_panel.cpp
  src/panels/fdm_panel.cpp
  src/panels/lob_panel.cpp
  src/gl/gl_loader.cpp
  src/gl/surface_renderer.cpp)
target_include_directories(quantviz_viz PRIVATE src)
target_link_libraries(quantviz_viz PRIVATE
  quantviz::scenes quantviz::vizcore quantviz::warnings
  ${QUANTVIZ_IMGUI_TARGETS} ${QUANTVIZ_GLFW_TARGET} OpenGL::GL)
