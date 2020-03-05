cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(NotSoSimplePerceptron ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH  ../NotSoSimplePerceptron)

target_include_directories(NotSoSimplePerceptron PRIVATE ${ANN_INCLUDE_DIR})

add_custom_target(copy_target_NotSoSimplePerceptron ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../NotSoSimplePerceptron/resources
        COMMENT "Copy neded folder to compiling executable")

target_include_directories(NotSoSimplePerceptron PRIVATE ${ANN_INCLUDE_DIR}
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

target_link_libraries(NotSoSimplePerceptron opencv_core opencv_imgcodecs)

add_dependencies(NotSoSimplePerceptron copy_target_NotSoSimplePerceptron)