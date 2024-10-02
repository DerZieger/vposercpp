#pragma  once

#include <cnpy.h>
#include <torch/torch.h>

//converts numpy data file into at::tensors for torch
template<typename T>
static torch::Tensor totorch(cnpy::npz_t &n, const std::string &s, torch::ScalarType type,
                             torch::ScalarType type2) {
    cnpy::NpyArray v = n[s];
    std::vector<int64_t> shape;
    for (unsigned long i: v.shape) shape.push_back(static_cast<long>(i));
    T *dat = v.data<T>();
    torch::Tensor ret = torch::from_blob(dat, shape, type).to(type2).clone();
    return ret;
}

torch::Tensor matrot2aa(const torch::Tensor &pose_matrot);

//convert rotation matrix to quaternion
torch::Tensor rotation_matrix_to_quaternion(const torch::Tensor &rotmat, float eps=1e-6);

//convert quaternion to rotation matrix
torch::Tensor quaternion_to_angle_axis(const torch::Tensor &q);