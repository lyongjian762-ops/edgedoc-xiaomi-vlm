cmake_minimum_required(VERSION 3.20)
project(edgedoc_npu_resident LANGUAGES CXX)

set(ROCKCHIP_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../output/xiaomi_ocr_setup/rknn-llm"
    CACHE PATH "Rockchip rknn-llm source checkout")
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${ROCKCHIP_SOURCE}" rev-parse HEAD
    RESULT_VARIABLE revision_status OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT revision_status EQUAL 0 OR NOT revision STREQUAL "f7390530443bf84f0394255a449d7cbe81e69d1c")
    message(FATAL_ERROR "Expected Rockchip rknn-llm at f7390530443bf84f0394255a449d7cbe81e69d1c")
endif()

set(deploy "${ROCKCHIP_SOURCE}/examples/multimodal_model_demo/deploy")
set(rknn "${deploy}/3rdparty/librknnrt/Linux/librknn_api")
set(rkllm "${ROCKCHIP_SOURCE}/rkllm-runtime/Linux/librkllm_api")
set(OpenCV_DIR "${deploy}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV")
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(edgedoc_npu_resident main.cpp "${deploy}/src/image_enc.cc")
target_compile_features(edgedoc_npu_resident PRIVATE cxx_std_17)
target_include_directories(edgedoc_npu_resident PRIVATE "${deploy}/src" "${rknn}/include" "${rkllm}/include")
target_link_libraries(edgedoc_npu_resident PRIVATE ${OpenCV_LIBS}
    "${rknn}/aarch64/librknnrt.so" "${rkllm}/aarch64/librkllmrt.so")
target_link_options(edgedoc_npu_resident PRIVATE
    "LINKER:--allow-shlib-undefined,-Bsymbolic,--version-script,${deploy}/c_export.map")
set_target_properties(edgedoc_npu_resident PROPERTIES SKIP_BUILD_RPATH TRUE)
