//
// Created by tim on 22.06.21.
//

#ifndef STEREO_RECONSTRUCTION_BUNDLEADJUSTMENT_H
#define STEREO_RECONSTRUCTION_BUNDLEADJUSTMENT_H

#include <stdio.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "Eigen.h"

/**
 * NOTE: Taken from lecture, exercise 5
 * Helper methods for writing Ceres cost functions.
 */
template <typename T>
static inline void fillVector(const Vector3f& input, T* output) {
    output[0] = T(input[0]);
    output[1] = T(input[1]);
    output[2] = T(input[2]);
}

template <typename T>
static inline void fillVector(const T* const input, T* output){
    output[0] = input[0];
    output[1] = input[1];
    output[2] = input[2];
 }


/**
 * NOTE: Taken from lecture, exercise 5
 * Pose increment is only an interface to the underlying array (in constructor, no copy
 * of the input array is made).
 * Important: Input array needs to have a size of at least 6.
 */
template <typename T>
class PoseIncrement {
public:
    explicit PoseIncrement(T* const array) : m_array{ array } { }

    void setZero() {
        for (int i = 0; i < 6; ++i)
            m_array[i] = T(0);
    }

    T* getData() const {
        return m_array;
    }

    /**
     * Applies the pose increment onto the input point and produces transformed output point.
     * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
     * beforehand).
     */
    void apply(T* inputPoint, T* outputPoint) const {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const T* rotation = m_array;
        const T* translation = m_array + 3;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }

    /**
     * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
     * transformation 4x4 matrix.
     */
    static Matrix4f convertToMatrix(const PoseIncrement<double>& poseIncrement) {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        double* pose = poseIncrement.getData();
        double* rotation = pose;
        double* translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Matrix4f matrix;
        matrix.setIdentity();
        matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
        matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
        matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);

        return matrix;
    }

private:
    T* m_array;
};

class BundleAdjustmentConstraint {
public:
    BundleAdjustmentConstraint(const Vector3f& keypoint, const Matrix3f& intrinsics, const int idx) :
                               m_keypoint{keypoint}, m_intrinsics{intrinsics}, m_idx(idx){}

    bool operator()(const T* const vars, T* residuals) const {
        // get pose from vars
        PoseIncrement<T> poseIncrement = PoseIncrement<T>(const_cast<T* const>(vars));

        // get corresponding point from vars
        T point3D[3];
        fillVector(point3D, vars+6 + m_idx*3)

        // transform 3d point using current pose
        T transformedPoint3D[3]
        poseIncrement.apply(point3D, &transformedPoint3D);

        // use Eigen to backproject 3D point to 2D image plane
        Vector<T, 3, 1> transformedPointEigen{transformedPoint3D};
        transformedPointEigen = (m_intrinsics.cast<T>() * transformedPointEigen) / transformedPointEigen[2];

        // residual
        residuals[0] = T(m_keypoint[0]) - transformedPointEigen[0];
        residuals[1] = T(m_keypoint[0]) - transformedPointEigen[0];
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>( //FIXME TODO
                new BundleAdjustmentConstraint(sourcePoint, targetPoint, targetNormal, weight)
        );
    }


private:
    const Vector3f m_keypoint;          /* in homogenous coordinates */
    const Matrix3f m_intrinsics;
    const int m_idx;
};

#endif //STEREO_RECONSTRUCTION_BUNDLEADJUSTMENT_H
