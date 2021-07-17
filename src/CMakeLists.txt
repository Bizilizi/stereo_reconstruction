cmake_minimum_required(VERSION 3.6)

set( CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE )
project( stereo_reconstruction )
set( CMAKE_CXX_STANDARD 17 )

#############################################################################################
# Package Requirements
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIR})

set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")

set(Midd_SDK_INCLUDE_DIR ${LIBRARY_DIR}/MiddEval3/code/ CACHE PATH "Path to Middleburry SDK")
set(imageLib_INCLUDE_DIR ${LIBRARY_DIR}/MiddEval3/code/imageLib/ CACHE PATH "Path to imageLib inside Middlebury SDK")

#############################################################################################
# Header / Source Files


file(GLOB HEADERS
        Eigen.h
        utils.h
        DataLoader/*.h
        Reconstruction/*.h
        PoseEstimation/*.h
        Rectification/*.h
        WindowSearch/*.h
        )

file(GLOB SOURCES
        DataLoader/*.cpp
        PoseEstimation/*.cpp
        Reconstruction/*.cpp
        Rectification/*.cpp
        WindowSearch/*.cpp
        utils.cpp
        )


#############################################################################################
# Executables

add_executable(test_loading Tests/test_loading.cpp ${HEADERS} ${SOURCES})
add_executable(test_keypoints Tests/test_keypoints.cpp ${HEADERS} ${SOURCES})
add_executable(test_bundle_adjustment Tests/test_bundle_adjustment.cpp ${HEADERS} ${SOURCES})
add_executable(test_reconstruction Tests/test_reconstruction.cpp ${HEADERS} ${SOURCES})
add_executable(rectification rectification_main.cpp ${HEADERS} ${SOURCES})

#############################################################################################
# Linking

# OpenCV
target_link_libraries(test_loading ${OpenCV_LIBS} Eigen3::Eigen)
target_link_libraries(test_keypoints ${OpenCV_LIBS} Eigen3::Eigen)
target_link_libraries(test_bundle_adjustment ${OpenCV_LIBS} Eigen3::Eigen)
target_link_libraries(test_reconstruction ${OpenCV_LIBS} Eigen3::Eigen)
target_link_libraries(rectification ${OpenCV_LIBS} Eigen3::Eigen)

# Eigen / imageLib
target_include_directories(test_loading PUBLIC ${imageLib_INCLUDE_DIR})
target_include_directories(test_keypoints PUBLIC ${imageLib_INCLUDE_DIR})
target_include_directories(test_bundle_adjustment PUBLIC ${imageLib_INCLUDE_DIR})
target_include_directories(test_reconstruction PUBLIC ${imageLib_INCLUDE_DIR})
target_include_directories(rectification PUBLIC ${imageLib_INCLUDE_DIR})

if (APPLE)
    target_link_libraries(rectification ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.i386-g.a )
    target_link_libraries(test_loading ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.i386-g.a )
    target_link_libraries(test_keypoints ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.i386-g.a )
    target_link_libraries(test_bundle_adjustment ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.i386-g.a )
    target_link_libraries(test_reconstruction ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.i386-g.a )
elseif(UNIX)
    target_link_libraries(rectification ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.x86_64-g.a )
    target_link_libraries(test_loading ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.x86_64-g.a )
    target_link_libraries(test_keypoints ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.x86_64-g.a )
    target_link_libraries(test_bundle_adjustment ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.x86_64-g.a )
    target_link_libraries(test_reconstruction ${LIBRARY_DIR}/MiddEval3/code/imageLib/libImg.x86_64-g.a )
endif()


# Ceres
target_link_libraries(test_bundle_adjustment ceres)
target_link_libraries(test_keypoints ceres)




