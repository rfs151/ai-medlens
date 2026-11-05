cmake_minimum_required(VERSION 3.16)
project(medlens_volume LANGUAGES CXX)

# One code base, two targets:
#   desktop:  cmake -B build && cmake --build build            -> build/medlens_volume (GLFW + OpenGL 3.3)
#   web:      emcmake cmake -B build-web && cmake --build build-web
#             -> medlens_volume.js + .wasm (WebGL2), copied to ../frontend/gl/

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CORE_SOURCES src/gl_api.cc src/gl_objects.cc src/volume.cc src/viewer.cc)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

if(EMSCRIPTEN)
  add_executable(medlens_volume ${CORE_SOURCES} src/main_web.cc)
  set_target_properties(medlens_volume PROPERTIES SUFFIX ".js")
  target_link_options(medlens_volume PRIVATE
    -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2
    -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB
    -sMODULARIZE=1 -sEXPORT_NAME=createMedlensVolume -sENVIRONMENT=web
    -sEXPORTED_FUNCTIONS=_main,_malloc,_free,_mv_load_volume,_mv_set_int,_mv_set_float,_mv_reset_view,_mv_state_json
    -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,UTF8ToString
    -sFILESYSTEM=0)
  # publish next to the web page so the Python service serves it as /gl/...
  add_custom_command(TARGET medlens_volume POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/../frontend/gl
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE_DIR:medlens_volume>/medlens_volume.js
            $<TARGET_FILE_DIR:medlens_volume>/medlens_volume.wasm ${CMAKE_SOURCE_DIR}/../frontend/gl/)
else()
  find_package(glfw3 3.3 REQUIRED)
  find_package(ZLIB REQUIRED)
  find_package(OpenGL REQUIRED)  # headers only: GL entry points are loaded at runtime (gl_api.cc)
  add_executable(medlens_volume ${CORE_SOURCES} src/main_native.cc)
  target_include_directories(medlens_volume PRIVATE ${OPENGL_INCLUDE_DIR})
  target_link_libraries(medlens_volume PRIVATE glfw ZLIB::ZLIB)
endif()
