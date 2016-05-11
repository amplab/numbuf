cmake_minimum_required(VERSION 2.8)

project(numbuf)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (UNIX AND NOT APPLE)
   link_libraries(rt)
endif()

set(ARROW_INCLUDE_DIR "" CACHE FILEPATH "Path to arrow includes")
include_directories(SYSTEM ${ARROW_INCLUDE_DIR})
set(ARROW_LIBRARY_DIR "" CACHE FILEPATH "Path to arrow library")
add_library(arrow SHARED IMPORTED)
set_target_properties(arrow PROPERTIES IMPORTED_LOCATION "${ARROW_LIBRARY_DIR}")

add_library(numbuf SHARED src/numbuf/tensor.cc src/numbuf/types.cc src/numbuf/metadata.cc src/numbuf/dict.cc)
target_link_libraries(numbuf arrow)

add_executable(test src/numbuf/test.cc)
target_link_libraries(test arrow)
target_link_libraries(test numbuf)

install(TARGETS numbuf
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  src/numbuf/dict.h
  src/numbuf/tensor.h
  src/numbuf/types.h
  src/numbuf/metadata.h
  DESTINATION include/numbuf/)

install(FILES
  src/numbuf/util/buffer.h
  src/numbuf/util/logging.h
  src/numbuf/util/utils.h
  DESTINATION include/numbuf/util/)
