cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(SimplePerceptron ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH  ../SimplePerceptron)

target_include_directories(SimplePerceptron PRIVATE ${ANN_INCLUDE_DIR})

add_custom_target(copy_target_SimplePerceptron ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../SimplePerceptron/resources
        COMMENT "Copy neded folder to compiling executable")

target_include_directories(SimplePerceptron PRIVATE ${ANN_INCLUDE_DIR}
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

target_link_libraries(SimplePerceptron opencv_core opencv_imgcodecs)

add_dependencies(SimplePerceptron copy_target_SimplePerceptron)