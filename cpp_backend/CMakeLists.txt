cmake_minimum_required(VERSION 3.16)
project(medlens_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Drogon CONFIG REQUIRED)
find_package(OpenCV 4.7 QUIET COMPONENTS core imgproc imgcodecs dnn)
if(NOT OpenCV_FOUND)
  # Debian/Ubuntu ship OpenCVConfig.cmake only in the full libopencv-dev (pulls GUI/video/Qt);
  # with just libopencv-{core,imgproc,imgcodecs,dnn}-dev, locate headers and libraries directly.
  find_path(OpenCV_INCLUDE_DIRS opencv2/dnn.hpp PATH_SUFFIXES opencv4 REQUIRED)
  file(STRINGS "${OpenCV_INCLUDE_DIRS}/opencv2/core/version.hpp" _cv_ver REGEX "#define CV_VERSION_(MAJOR|MINOR) ")
  string(REGEX REPLACE ".*MAJOR +([0-9]+).*MINOR +([0-9]+).*" "\\1.\\2" OpenCV_VERSION "${_cv_ver}")
  if(OpenCV_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "OpenCV >= 4.7 required (YOLOv8/11 ONNX + NMSBoxesBatched), found ${OpenCV_VERSION}")
  endif()
  set(OpenCV_LIBS)
  foreach(m core imgproc imgcodecs dnn)
    find_library(OpenCV_${m}_LIB opencv_${m} REQUIRED)
    list(APPEND OpenCV_LIBS ${OpenCV_${m}_LIB})
  endforeach()
endif()
message(STATUS "OpenCV ${OpenCV_VERSION}: ${OpenCV_INCLUDE_DIRS}")

add_executable(medlens_server src/main.cc src/yolo_detector.cc)
target_include_directories(medlens_server PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(medlens_server PRIVATE Drogon::Drogon ${OpenCV_LIBS})

# Optional micro-benchmark without HTTP: ./bench_detector model.onnx image.jpg [runs]
add_executable(bench_detector src/bench.cc src/yolo_detector.cc)
target_include_directories(bench_detector PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bench_detector PRIVATE Drogon::Drogon ${OpenCV_LIBS})
