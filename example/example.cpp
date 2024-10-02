#include "../vposercpp.h"
#include "fstream"

int main(){
    auto a=VPoserCPP("/EXAMPLE_PATH/vposer.npz");
    auto in = torch::zeros({1,a.getLatentD()});
    auto b=a.Decode(in);
    std::cout<<b["pose_body"]<<std::endl;
    return 0;
}