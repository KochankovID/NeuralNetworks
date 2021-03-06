cmake_minimum_required(VERSION 3.0.0)

file(GLOB_RECURSE TARGET_SRC "./src/*.cpp")

add_executable(Test ${TARGET_SRC})

enable_testing()

if (${CMAKE_SYSTEM} MATCHES "Windows")
    target_link_libraries(Test PRIVATE gtest gtest_main NeuralNetworks)
    set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
else()
    target_link_libraries(Test PRIVATE gtest gtest_main pthread NeuralNetworks)
endif ()

add_subdirectory(lib/googletest)

target_include_directories(Test PRIVATE ${GTEST_INCLUDE_DIR})

add_test(AllTestsInMain Test)

set(EXECUTABLE_OUTPUT_PATH  ../Tests)