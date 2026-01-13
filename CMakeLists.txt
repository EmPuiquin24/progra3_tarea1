cmake_minimum_required(VERSION 3.29)
project(TensorPP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB_RECURSE SOURCES "src/*.cpp")

include_directories(include)

add_executable(${PROJECT_NAME} ${SOURCES})
