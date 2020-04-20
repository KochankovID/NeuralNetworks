cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(P_Network_one ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH .)

target_include_directories(P_Network_one PRIVATE ${ANN_INCLUDE_DIR})
find_package (Boost COMPONENTS filesystem REQUIRED)

add_custom_target(copy_target_P_Network ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources resources
        COMMENT "Copy neded folder to compiling executable")

target_include_directories(P_Network_one PRIVATE ${ANN_INCLUDE_DIR}
        "../lib/opencv/include"
        "../lib/opencv/modules/core/include"
        "../lib/opencv/modules/ts/include"
        "../lib/opencv/modules/highgui/include"
        "../lib/opencv/modules/imgcodecs/include"
        "../lib/opencv/modules/videoio/include"
        "../lib/opencv/modules/imgproc/include"
        "../lib/opencv/modules"
        "../lib/fast-cpp-csv-parser"
        "${CMAKE_BINARY_DIR}"
        "${Boost_INCLUDE_DIRS}"
        )

target_link_libraries(P_Network_one opencv_core opencv_imgcodecs)
target_link_libraries(P_Network_one ${Boost_LIBRARIES})

add_dependencies(P_Network_one copy_target_P_Network)