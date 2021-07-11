cmake_minimum_required(VERSION 3.6)

set( CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE )
project( stereo_reconstruction )
set( CMAKE_CXX_STANDARD 17 )

#############################################################################################
# Package Requirements
find_package( Ceres REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/eigen-3.4/ CACHE PATH "Path to Eigen source folder")

set(Midd_SDK_INCLUDE_DIR ${LIBRARY_DIR}/MiddEval3/code/ CACHE PATH "Path to Middleburry SDK")
set(imageLib_INCLUDE_DIR ${LIBRARY_DIR}/MiddEval3/code/imageLib/ CACHE PATH "Path to imageLib inside Middlebury SDK")

#############################################################################################
# Header / Source Files

set(HEADERS
    Eigen.h
    directory.h
    eight_point.h
    SimpleMesh.h
    BundleAdjustment.h
    data_loader.h
    data.h)

set(SOURCES
    directory.cpp eight_point.cpp data_loader.cpp data.cpp)

#############################################################################################
# Executables

add_executable(test_loading test_loading.cpp ${HEADERS} ${SOURCES})
add_executable(keypoints keypoints.cpp ${HEADERS} ${SOURCES})
add_executable(test_bundle_adjustment test_bundle_adjustment.cpp ${HEADERS} ${SOURCES})


#############################################################################################
# Linking

#target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(test_loading ${OpenCV_LIBS})
target_link_libraries(keypoints ${OpenCV_LIBS})
target_link_libraries(test_bundle_adjustment ${OpenCV_LIBS})

#target_include_directories(main PUBLIC ${Eigen3_INCLUDE_DIR})
target_include_directories(test_loading PUBLIC ${Eigen3_INCLUDE_DIR} ${imageLib_INCLUDE_DIR})
target_include_directories(keypoints PUBLIC ${Eigen3_INCLUDE_DIR})


target_include_directories(test_bundle_adjustment PUBLIC ${Eigen3_INCLUDE_DIR} )

target_link_libraries(test_bundle_adjustment ceres)
target_link_libraries(test_loading ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.x86_64-g.a )

target_link_libraries(keypoints ceres)
#target_link_libraries(surface ceres)
