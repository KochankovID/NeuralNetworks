cmake_minimum_required(VERSION 3.0.0)

file(GLOB_RECURSE TARGET_SRC "./src/*.cpp")

add_executable(Test ${TARGET_SRC} src/Tests_Filter/test_Filter.cpp src/Tests_CnnLearning/test_CnnLearning.cpp src/Tests_PerceptronLearning/test_PerceptronLearning.cpp src/Tests_Matrix/test_Matrix.cpp src/Tests_Perceptron/test_Perceptron.cpp src/Tests_CnnLearning/perf_test_CnnLearning.cpp src/Tests_Filter/perf_test_Filter.cpp src/Tests_Matrix/perf_test_Matrix.cpp src/Tests_Perceptron/perf_test_Perceptron.cpp src/Tests_PerceptronLearning/perf_test_PerceptronLearning.cpp)

target_include_directories(Test PRIVATE ${ANN_INCLUDE_DIR}
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

target_link_libraries(Test opencv_ts)

set(EXECUTABLE_OUTPUT_PATH  ../Tests)