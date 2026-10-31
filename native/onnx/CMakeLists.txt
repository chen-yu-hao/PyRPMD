cmake_minimum_required(VERSION 3.18)
project(pyrpmd_onnx LANGUAGES CXX)
# Optional standalone plugin; the ordinary PyRPMD build never requires ORT.
set(ONNXRUNTIME_ROOT "" CACHE PATH "Unpacked ONNX Runtime C/C++ SDK")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}" ENV ONNXRUNTIME_ROOT
  PATH_SUFFIXES include include/onnxruntime/core/session REQUIRED)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}" ENV ONNXRUNTIME_ROOT PATH_SUFFIXES lib lib64 REQUIRED)
find_package(nlohmann_json 3.10 REQUIRED)
add_library(pyrpmd_onnx SHARED onnx_pes.cpp)
target_compile_features(pyrpmd_onnx PRIVATE cxx_std_17)
target_include_directories(pyrpmd_onnx PRIVATE ../include "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(pyrpmd_onnx PRIVATE "${ONNXRUNTIME_LIBRARY}" nlohmann_json::nlohmann_json)
set_target_properties(pyrpmd_onnx PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pyrpmd_onnx PRIVATE -Wall -Wextra -Wpedantic)
endif()
# Test a native multi-context caller, independently of Python's conservative
# per-library lock. This does NOT replace PyRPMD's trajectory scheduler.
option(PYRPMD_ONNX_CONTEXT_TEST "Build the OpenMP context-isolation test" OFF)
if(PYRPMD_ONNX_CONTEXT_TEST)
  find_package(OpenMP REQUIRED)
  add_executable(onnx_context_test context_test.cpp)
  target_compile_features(onnx_context_test PRIVATE cxx_std_17)
  target_include_directories(onnx_context_test PRIVATE ../include)
  target_link_libraries(onnx_context_test PRIVATE pyrpmd_onnx OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json)
endif()
install(TARGETS pyrpmd_onnx LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
