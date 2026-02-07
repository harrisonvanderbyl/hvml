cmake_minimum_required(VERSION 3.16)
project(shader_tool)

find_package(Clang REQUIRED CONFIG)

add_executable(shader_tool tool.cpp)

target_include_directories(shader_tool PRIVATE
    ${CLANG_INCLUDE_DIRS}
)

target_link_libraries(shader_tool PRIVATE clang-cpp LLVM)

