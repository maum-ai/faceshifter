import argparse
from PIL import Image
from omegaconf import OmegaConf

import torch
from torchvision import transforms

from aei_net import AEINet

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                    help="path of configuration yaml file")
parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="path of aei-net pre-trained file")
parser.add_argument("--target_image", type=str, required=True,
                    help="path of preprocessed target face image")
parser.add_argument("--source_image", type=str, required=True,
                    help="path of preprocessed source face image")
parser.add_argument("--output_path", type=str, default="output.png",
                    help="path of output image")
parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu")
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

hp = OmegaConf.load(args.config)
model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=hp)
model.eval()
model.freeze()
model.to(device)

target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)
source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)
with torch.no_grad():
    output, _, _, _, _ = model.forward(target_img, source_img)
output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
output.save(args.output_path)
