
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

void reconstruction(cv::Mat bgrImage, cv::Mat depthValues, Matrix3f intrinsics, float thrMesh=1.0f);

cv::Mat convertDisparityToDepth(const cv::Mat& dispImage, float focalLength, float baseline=1.0f);

void scaleDisparityMap(cv::Mat& disparityMap, float scalingFactor);

void removeDisparityOutliers(cv::Mat& disparityMap, int kernelSize=500, float thrFront=1.5, float thrBack=0.8);

#endif //STEREO_RECONSTRUCTION_RECONSTRUCTION_H
