cmake_minimum_required(VERSION 3.20)
project(edgedoc_cam LANGUAGES C CXX)

set(EDGEDOC_LLAMA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../output/xiaomi_ocr_setup/llama.cpp"
    CACHE PATH "Source checkout of the pinned llama.cpp revision")
set(EDGEDOC_LLAMA_COMMIT 8642541b255bfb5dc39cd351a178be09013a55d6)
find_package(Git REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${EDGEDOC_LLAMA_SOURCE}" rev-parse --verify HEAD
    RESULT_VARIABLE revision_status OUTPUT_VARIABLE actual_revision OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT revision_status EQUAL 0 OR NOT actual_revision STREQUAL EDGEDOC_LLAMA_COMMIT)
    message(FATAL_ERROR "EDGEDOC_LLAMA_SOURCE must point to llama.cpp at ${EDGEDOC_LLAMA_COMMIT}")
endif()

set(LLAMA_BUILD_COMMON ON)
set(LLAMA_BUILD_MTMD ON)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_BUILD_SERVER OFF)
set(LLAMA_BUILD_APP OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(MTMD_VIDEO OFF CACHE BOOL "Enable video decoding in libmtmd" FORCE)
add_subdirectory("${EDGEDOC_LLAMA_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

add_executable(edgedoc_cli src/edgedoc_cli.cpp)
target_compile_features(edgedoc_cli PRIVATE cxx_std_17)
target_link_libraries(edgedoc_cli PRIVATE mtmd llama-common)
target_compile_definitions(edgedoc_cli PRIVATE EDGEDOC_LLAMA_COMMIT="${EDGEDOC_LLAMA_COMMIT}")
