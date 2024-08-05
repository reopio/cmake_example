cmake_minimum_required(VERSION 3.18)

project(test CXX)

include_directories(./include)

find_library(LIB2 liblb2.a ./lib)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(MYLIB STATIC ${SOURCES})

add_executable(t ./main.cpp)

target_link_libraries(t PRIVATE MYLIB)
target_link_libraries(t PRIVATE ${LIB2})