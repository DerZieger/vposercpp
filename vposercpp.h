#pragma once
#include <torch/types.h>
#include <torch/nn.h>
#include <unordered_map>
#include "vputil.h"


//Wrapper around libtorch's normal distribution function to have a class
class NormalDistribution {
public:
    torch::Tensor loc, scale;

    NormalDistribution(torch::Tensor loc, torch::Tensor scale);

    torch::Tensor rsample() const;
};

//Libtorch Module that flattens out a tensor's dimensions after the first
struct BatchFlatten : public torch::nn::Module {
public:
    BatchFlatten() = default;

    torch::Tensor forward(const torch::Tensor &input);
};

//Libtorch Module that decodes the input as continuous rotation
struct ContinousRotReprDecoder : public torch::nn::Module {
public:
    ContinousRotReprDecoder() = default;

    torch::Tensor forward(const torch::Tensor &input);
};

//Libtorch module that returns a normal distributions wrapper for the given input
struct NormalDistDecoder : public torch::nn::Module {
    torch::nn::Linear mu{nullptr}, logvar{nullptr};

public:
    NormalDistDecoder(int num_feat_in, int latentD);

    NormalDistribution forward(const torch::Tensor &input);
};

class VPoserCPP : public torch::nn::Module{
public:
    VPoserCPP(std::string path, int num_joints = 21, int repr = 3);

    NormalDistribution Encode(const torch::Tensor &body_pose);

    std::unordered_map<std::string,torch::Tensor> Decode(const torch::Tensor &zin);

    std::unordered_map<std::string,torch::Tensor> Forward(const torch::Tensor &body_pose);

    int getLatentD() const;

private:
    int m_num_features, m_rotrepr, m_latentD;
    torch::nn::Sequential m_encoder, m_decoder;
};
