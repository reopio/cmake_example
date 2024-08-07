cmake_minimum_required(VERSION 3.18)

project(test CUDA CXX)

include_directories(./include)

find_package(CUDA REQUIRED)
include_directories($(CUDA_INCLUDE_DIRS))

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(MYLIB STATIC ${SOURCES})

add_executable(t ./main.cu)

target_link_libraries(t PRIVATE MYLIB)
target_link_libraries(t PRIVATE ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
