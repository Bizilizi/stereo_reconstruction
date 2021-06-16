cmake_minimum_required(VERSION 3.6)

set( CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE )
project( stereo_reconstruction )
set( CMAKE_CXX_STANDARD 14 )


#############################################################################################
# Package Requirements
find_package( Ceres REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/eigen-3.4/ CACHE PATH "Path to Eigen source folder")

#############################################################################################
# Header / Source Files

set(HEADERS
        Eigen.h)

# same for Sources later


#############################################################################################
# Executables



add_executable(main main.cpp ${HEADERS} )

add_executable(test_loading test_loading.cpp ${HEADERS} )


#############################################################################################
# Linking

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(test_loading ${OpenCV_LIBS})
target_include_directories(main PUBLIC ${Eigen3_INCLUDE_DIR})
target_include_directories(test_loading PUBLIC ${Eigen3_INCLUDE_DIR})

#target_link_libraries(dragon ceres)
#target_link_libraries(surface ceres)

