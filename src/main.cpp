
#include <iostream>
//#include <opencv4/opencv2/opencv.hpp>
#include "Eigen.h"

int main()
{
    std::cout << "Hello world! " << std::endl;
    //cv::Mat image;
    MatrixXf testMat = Matrix3f::Zero();
    std::cout << testMat;
    return 0;
}