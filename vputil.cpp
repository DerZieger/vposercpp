#include "vputil.h"


torch::Tensor toTorchFloating(cnpy::npz_t &n, const std::string &s) {
    if (n[s].word_size == 4) {
        return totorch<float>(n, s, torch::kFloat32, torch::kFloat32);
    } else {
        return totorch<double>(n, s, torch::kFloat64, torch::kFloat32);
    }
}

torch::Tensor toTorchInt(cnpy::npz_t &n, const std::string &s, torch::ScalarType type) {
    if (n[s].word_size == 4) {
        return totorch<int32_t>(n, s, torch::kInt32, type);
    } else {
        return totorch<int64_t>(n, s, torch::kInt64, type);
    }
}

torch::Tensor matrot2aa(const torch::Tensor &pose_matrot) {
    torch::Tensor homogen_matrot = torch::nn::functional::pad(pose_matrot, torch::nn::functional::PadFuncOptions({0, 1}));
    return quaternion_to_angle_axis(rotation_matrix_to_quaternion(homogen_matrot));
}

torch::Tensor quaternion_to_angle_axis(const torch::Tensor &q) {
    long batch_size = q.sizes()[0];
    torch::Tensor p1 = q.index({"...", 1});
    torch::Tensor p2 = q.index({"...", 2});
    torch::Tensor p3 = q.index({"...", 3});
    torch::Tensor sin_squared_theta = p1 * p1 + p2 * p2 + p3 * p3;

    torch::Tensor sin_theta = torch::sqrt(sin_squared_theta);
    torch::Tensor cos_theta = q.index({"...", 0});
    torch::Tensor two_theta =
            2.0 *
            torch::where(cos_theta < 0.0, torch::atan2(-sin_theta, -cos_theta), torch::atan2(sin_theta, cos_theta));

    torch::Tensor k_pos = two_theta / sin_theta;
    torch::Tensor k_neg = 2.0 * torch::ones_like(sin_theta);
    torch::Tensor k = torch::where(sin_squared_theta > 0.0, k_pos, k_neg);

    using namespace torch::indexing;
    torch::Tensor angle_axis = torch::zeros_like(q).index({"...", Slice(None, 3)});
    angle_axis.index({"...", 0}) += p1 * k;
    angle_axis.index({"...", 1}) += p2 * k;
    angle_axis.index({"...", 2}) += p3 * k;


    return angle_axis.view({batch_size, -1, 3});
}

torch::Tensor rotation_matrix_to_quaternion(const torch::Tensor &rotmat, float eps) {
    torch::Tensor rmat_t = torch::transpose(rotmat, 1, 2);

    using namespace torch::indexing;

    torch::Tensor rmat_t_0_0 = rmat_t.index({Slice(), 0, 0});
    torch::Tensor rmat_t_1_1 = rmat_t.index({Slice(), 1, 1});
    torch::Tensor rmat_t_2_2 = rmat_t.index({Slice(), 2, 2});
    torch::Tensor rmat_t_0_1 = rmat_t.index({Slice(), 0, 1});
    torch::Tensor rmat_t_0_2 = rmat_t.index({Slice(), 0, 2});
    torch::Tensor rmat_t_1_0 = rmat_t.index({Slice(), 1, 0});
    torch::Tensor rmat_t_1_2 = rmat_t.index({Slice(), 1, 2});
    torch::Tensor rmat_t_2_0 = rmat_t.index({Slice(), 2, 0});
    torch::Tensor rmat_t_2_1 = rmat_t.index({Slice(), 2, 1});

    torch::Tensor mask_d2 = rmat_t.index({Slice(), 2, 2}) < eps;

    torch::Tensor mask_d0_d1 = rmat_t_0_0 > rmat_t_1_1;
    torch::Tensor mask_d0_nd1 = rmat_t_0_0 < -rmat_t_1_1;

    torch::Tensor t0 = 1 + rmat_t_0_0 - rmat_t_1_1 - rmat_t_2_2;
    torch::Tensor q0 = torch::stack({rmat_t_1_2 - rmat_t_2_1, t0, rmat_t_0_1 + rmat_t_1_0, rmat_t_2_0 + rmat_t_0_2},
                                    -1);
    torch::Tensor t0_rep = t0.repeat(torch::IntArrayRef({4, 1})).t();

    torch::Tensor t1 = 1 - rmat_t_0_0 + rmat_t_1_1 - rmat_t_2_2;
    torch::Tensor q1 = torch::stack({rmat_t_2_0 - rmat_t_0_2, rmat_t_0_1 + rmat_t_1_0, t1, rmat_t_1_2 + rmat_t_2_1},
                                    -1);
    torch::Tensor t1_rep = t1.repeat(torch::IntArrayRef({4, 1})).t();

    torch::Tensor t2 = 1 - rmat_t_0_0 - rmat_t_1_1 + rmat_t_2_2;
    torch::Tensor q2 = torch::stack({rmat_t_0_1 - rmat_t_1_0, rmat_t_2_0 + rmat_t_0_2, rmat_t_1_2 + rmat_t_2_1, t2},
                                    -1);
    torch::Tensor t2_rep = t2.repeat(torch::IntArrayRef({4, 1})).t();

    torch::Tensor t3 = 1 + rmat_t_0_0 + rmat_t_1_1 + rmat_t_2_2;
    torch::Tensor q3 = torch::stack({t3, rmat_t_1_2 - rmat_t_2_1, rmat_t_2_0 - rmat_t_0_2, rmat_t_0_1 - rmat_t_1_0},
                                    -1);
    torch::Tensor t3_rep = t3.repeat(torch::IntArrayRef({4, 1})).t();

    torch::Tensor mask_c0 = mask_d2 * mask_d0_d1;
    torch::Tensor mask_c1 = mask_d2 * torch::logical_not(mask_d0_d1);
    torch::Tensor mask_c2 = torch::logical_not(mask_d2) * mask_d0_nd1;
    torch::Tensor mask_c3 = torch::logical_not(mask_d2) * torch::logical_not(mask_d0_nd1);
    mask_c0 = mask_c0.view({-1, 1}).type_as(q0);
    mask_c1 = mask_c1.view({-1, 1}).type_as(q1);
    mask_c2 = mask_c2.view({-1, 1}).type_as(q2);
    mask_c3 = mask_c3.view({-1, 1}).type_as(q3);

    torch::Tensor q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3;
    q /= torch::sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3);
    q *= 0.5;

    return q;
}

