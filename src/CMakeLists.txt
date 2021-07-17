cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
project(stereo_reconstruction)
set(CMAKE_CXX_STANDARD 17)

#############################################################################################
# Package Requirements
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIR})

set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")

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
        data.h
        data_loader.h
        stero_matching/LinearSearch.h
        stero_matching/BlockSearch.h
        rectification/rectification.hpp)

set(SOURCES
        directory.cpp
        eight_point.cpp
        stero_matching/LinearSearch.cpp
        stero_matching/BlockSearch.cpp
        rectification/rectification.cpp
        data.cpp
        data_loader.cpp)

#############################################################################################
# Executables

add_executable(test_loading test_loading.cpp ${HEADERS} ${SOURCES})
add_executable(rectification rectification_main.cpp ${HEADERS} ${SOURCES} evalDisp.h)
add_executable(test_bundle_adjustment test_bundle_adjustment.cpp ${HEADERS} ${SOURCES})


#############################################################################################
# Linking
target_include_directories(rectification PUBLIC ${imageLib_INCLUDE_DIR})
if (APPLE)
    target_link_libraries(rectification ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.i386-g.a)
elseif(UNIX)
    target_link_libraries(rectification ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.x86_64-g.a )
endif (MACOS)


#target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(test_loading ${OpenCV_LIBS})
target_link_libraries(rectification ${OpenCV_LIBS})
target_link_libraries(test_bundle_adjustment ${OpenCV_LIBS})

#target_include_directories(main PUBLIC ${Eigen3_INCLUDE_DIR})
target_link_libraries(test_loading Eigen3::Eigen)
target_link_libraries(rectification Eigen3::Eigen)
target_link_libraries(test_bundle_adjustment Eigen3::Eigen)

target_link_libraries(test_bundle_adjustment ceres)
#target_link_libraries(dragon ceres)
#target_link_libraries(surface ceres)
