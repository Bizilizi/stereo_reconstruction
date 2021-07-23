
#ifndef STEREO_RECONSTRUCTION_RECONSTRUCTION_H
#define STEREO_RECONSTRUCTION_RECONSTRUCTION_H

#include <iostream>
#include <fstream>
#include <array>
#include <opencv4/opencv2/opencv.hpp>

#include "../Eigen.h"


struct Vertex
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f position;

    // color stored as 4 unsigned char
    Vector4uc color;
};

bool CheckTriangularValidity(Vertex* vertices, unsigned int one, unsigned int two, unsigned int three, float threshold);

bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename, float edgeThreshold);

void reconstruction(cv::Mat bgrImage, cv::Mat depthValues, Matrix3f intrinsics, float thrMarchingSquares=1.0f);

cv::Mat convertDispartiyToDepth(const cv::Mat& dispImage, float focalLength, float baseline=1.0f);

#endif //STEREO_RECONSTRUCTION_RECONSTRUCTION_H
