cmake_minimum_required(VERSION 3.0.0)

file(GLOB_RECURSE TARGET_SRC "./src/*.cpp")

add_executable(Perf_test ${TARGET_SRC})

target_include_directories(Perf_test PRIVATE ${ANN_INCLUDE_DIR}
        "../lib/opencv/include"
        "../lib/opencv/modules/core/include"
        "../lib/opencv/modules/ts/include"
        "../lib/opencv/modules/highgui/include"
        "../lib/opencv/modules/imgcodecs/include"
        "../lib/opencv/modules/videoio/include"
        "../lib/opencv/modules/imgproc/include"
        "../lib/opencv/modules"
        "${CMAKE_BINARY_DIR}"
        )

target_link_libraries(Perf_test opencv_ts)

set(EXECUTABLE_OUTPUT_PATH  ../Perf_tests)