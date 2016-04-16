cmake_minimum_required(VERSION 2.8)

project(ether)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

include_directories("${PYTHON_INCLUDE_DIRS}")
include_directories("${NUMPY_INCLUDE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (UNIX AND NOT APPLE)
   link_libraries(rt)
endif()

add_library(ether SHARED src/ether.cc src/serialize.cc src/array.cc src/dict.cc)
target_link_libraries(ether arrow)

install(TARGETS ether
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES src/api.h DESTINATION include/ether/)
