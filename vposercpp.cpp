#include "vposercpp.h"
#include "vputil.h"
#include <filesystem>

VPoserCPP::VPoserCPP(std::string path, int num_joints, int repr) : m_num_features(repr * num_joints), m_rotrepr(repr) {
    std::map<std::string, cnpy::NpyArray> data = cnpy::npz_load(std::move(path));
    //get sizes
    int num_neurons = data["num_neurons"].data<int>()[0];
    m_latentD = data["latentD"].data<int>()[0];
    //setup vae
    m_encoder = torch::nn::Sequential(BatchFlatten(), torch::nn::BatchNorm1d(m_num_features),
                                      torch::nn::Linear(m_num_features, num_neurons),
                                      torch::nn::LeakyReLU(), torch::nn::BatchNorm1d(num_neurons),
                                      torch::nn::Dropout(0.1),
                                      torch::nn::Linear(num_neurons, num_neurons),
                                      torch::nn::Linear(num_neurons, num_neurons),
                                      NormalDistDecoder(num_neurons, m_latentD));

    m_decoder = torch::nn::Sequential(torch::nn::Linear(m_latentD, num_neurons), torch::nn::LeakyReLU(),
                                      torch::nn::Dropout(0.1),
                                      torch::nn::Linear(num_neurons, num_neurons), torch::nn::LeakyReLU(),
                                      torch::nn::Linear(num_neurons, num_joints * 6), ContinousRotReprDecoder());


    this->register_module("encoder_net", m_encoder);
    this->register_module("decoder_net", m_decoder);


    using namespace torch::indexing;
    for (auto &m: this->named_parameters()) {
        if (data.count(m.key())) {
            m.value().requires_grad_(false);//Just inference
            m.value().copy_(totorch<float>(data, m.key(), torch::kFloat32, torch::kFloat32));//load pretrained vposer
        }
    }
    for (torch::OrderedDict<std::string, torch::Tensor>::Item &m: this->named_buffers()) {
        if (data.count(m.key())) {
            m.value().requires_grad_(false);//Just inference
            std::string end = "num_batches_tracked";
            if (m.key().length() >= end.length() && (0 == m.key().compare(m.key().length() - end.length(), end.length(),
                                                                          end))) {//compare if string has end as ending from hasEnding Saiga
                m.value().copy_(
                        totorch<int64_t>(data, m.key(), torch::kInt64, torch::kInt64));//load pretrained vposer
            } else {
                m.value().copy_(
                        totorch<float>(data, m.key(), torch::kFloat32, torch::kFloat32));//load pretrained vposer
            }
        }
    }
    m_encoder->eval();
    m_decoder->eval();
}

NormalDistribution VPoserCPP::Encode(const torch::Tensor &body_pose) {
    return m_encoder->forward<NormalDistribution>(body_pose);
}

std::unordered_map<std::string, torch::Tensor> VPoserCPP::Decode(const torch::Tensor &zin) {
    int batch_size = zin.size(0);
    torch::Tensor prec = m_decoder->forward(zin);
    return {{"pose_body",        matrot2aa(prec.view({-1,3,3})).view({batch_size,-1,3})},
            {"pose_body_matrot", prec.view({batch_size,-1,9})}};
}

std::unordered_map<std::string, torch::Tensor> VPoserCPP::Forward(const torch::Tensor &body_pose) {
    NormalDistribution q_z = Encode(body_pose);
    torch::Tensor sample = q_z.rsample();
    std::unordered_map<std::string, torch::Tensor> decode_results = Decode(sample);
    decode_results["poZ_body_mean"]=q_z.loc;
    decode_results["poZ_body_std"]=q_z.scale;
    return decode_results;
}

int VPoserCPP::getLatentD()const{
    return m_latentD;
}


NormalDistribution::NormalDistribution(torch::Tensor loc, torch::Tensor scale) : loc(std::move(loc)), scale(std::move(scale)) {}

torch::Tensor NormalDistribution::rsample() const { return at::normal(loc, scale); }

torch::Tensor BatchFlatten::forward(const torch::Tensor &input) { return input.view({input.size(0), -1}); }

torch::Tensor ContinousRotReprDecoder::forward(const torch::Tensor &input) {
    using namespace torch::indexing;
    torch::Tensor reshaped = input.view({-1, 3, 2});
    namespace F = torch::nn::functional;
    torch::Tensor b1 = F::normalize(reshaped.index({Slice(), Slice(), 0}), F::NormalizeFuncOptions().dim(1));

    torch::Tensor dot = torch::sum(b1 * reshaped.index({Slice(), Slice(), 1}), 1, true);
    torch::Tensor b2 = F::normalize(reshaped.index({Slice(), Slice(), 1}) - dot * b1, F::NormalizeFuncOptions().dim(-1));
    torch::Tensor b3 = torch::cross(b1, b2, 1);
    return torch::stack({b1, b2, b3}, -1);
}

NormalDistDecoder::NormalDistDecoder(int num_feat_in, int latentD)
        : mu(torch::nn::Linear(num_feat_in, latentD)), logvar(torch::nn::Linear(num_feat_in, latentD)) {
    this->register_module("mu", mu);
    this->register_module("logvar", logvar);
}

NormalDistribution NormalDistDecoder::forward(const torch::Tensor &input) {
    return NormalDistribution(mu(input), torch::softplus(logvar(input)));
}