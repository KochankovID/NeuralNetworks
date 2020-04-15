cmake_minimum_required(VERSION 3.0.0)

add_executable(Image_to_txt ./src/Main.cpp)

set(EXECUTABLE_OUTPUT_PATH  ../Image_to_txt)
find_package (Boost COMPONENTS filesystem REQUIRED)

target_include_directories(Image_to_txt PRIVATE ${ANN_INCLUDE_DIR}
        "../lib/opencv/include"
        "../lib/opencv/modules/core/include"
        "../lib/opencv/modules/ts/include"
        "../lib/opencv/modules/highgui/include"
        "../lib/opencv/modules/imgcodecs/include"
        "../lib/opencv/modules/videoio/include"
        "../lib/opencv/modules/imgproc/include"
        "../lib/opencv/modules"
        "${CMAKE_BINARY_DIR}"
        "${Boost_INCLUDE_DIRS}"
        )

target_link_libraries(Image_to_txt ${Boost_LIBRARIES})
target_link_libraries(Image_to_txt opencv_core opencv_imgcodecs)

add_custom_target(copy_target_Image_to_txt ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../Image_to_txt/resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(Image_to_txt copy_target_Image_to_txt)

