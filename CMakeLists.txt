cmake_minimum_required(VERSION 3.28)
project(hwmn LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build tests" ON)

include(cmake/CPM.cmake)
include(cmake/third_party.cmake)

add_library(matrices STATIC)
target_sources(matrices
  PUBLIC
    FILE_SET CXX_MODULES
    FILES
      source/floating_point_comparing.cppm
      source/array.cppm
      source/jagged_array.cppm
      source/jagged_array_matrix.cppm
)
target_compile_features(matrices PUBLIC cxx_std_23)

add_executable(${PROJECT_NAME} source/main.cpp)
target_sources(${PROJECT_NAME}
  PRIVATE
    FILE_SET CXX_MODULES
    FILES
      source/util.cppm
)
target_link_libraries(${PROJECT_NAME} PRIVATE matrices)

foreach(target ${PROJECT_NAME})
  target_compile_options(${target}
    PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/Wall>
      $<$<CXX_COMPILER_ID:GNU,Clang>: -Wall -Wextra -pedantic>
  )
endforeach()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
