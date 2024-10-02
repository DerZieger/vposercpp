import numpy as np
import torch
import yaml
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p",type=str,default="./V02_5.yaml")
args = parser.parse_args()
path_to_yaml=args.p
vposer_dict={}
with open(path_to_yaml,"r") as file:
    cfg=yaml.safe_load(file)
    vposer_dict["latentD"]=cfg["model_params"]["latentD"]
    vposer_dict["num_neurons"]=cfg["model_params"]["num_neurons"]
for k,v in torch.load(str(Path(path_to_yaml).parent)+"/snapshots/V02_05_epoch=08_val_loss=0.03.ckpt", map_location=torch.device("cpu"))["state_dict"].items():
    if k.startswith("vp_model."):
        vposer_dict[k.replace("vp_model.", "")]=torch.Tensor.cpu(v)
np.savez("./vposer.npz",**vposer_dict)