import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

# from model.mnist import MNISTModel
from main_mnist_00_train import Net


#------------------------------------------------------------------------------

import argparse

from pytorch_nndct.apis import torch_quantizer
import torch


parser = argparse.ArgumentParser()

parser.add_argument(
    '--weights',
    default="../01_trained_model/runs/train/exp/weights/best.pt",
    help='.pt file')
parser.add_argument(
    '--imgsz',
    default=640,
    type=int,
    help='image size')

args, _ = parser.parse_known_args()

def attempt_load(weights, device=None, inplace=True, fuse=True):
    
    ckpt = torch.load(weights, map_location='cpu')  # load
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

    # Model compatibility updates
    if not hasattr(ckpt, 'stride'):
        ckpt.stride = torch.tensor([32.])
    if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
        ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

    ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval()  # model in eval mode

    # Module compatibility updates
    # for m in model.modules():
    #     t = type(m)
    #     if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
    #         m.inplace = inplace  # torch 1.7.0 compatibility
    #         if t is Detect and not isinstance(m.anchor_grid, list):
    #             delattr(m, 'anchor_grid')
    #             setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
    #     elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
    #         m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    return ckpt


# def warmup(model, imgsz=(1, 3, 640, 640)):
#     # Warmup model by running inference once
#     im = torch.empty(*imgsz, dtype=torch.float, device=device)  # input
#     for _ in range(2 if self.jit else 1):  #
#         y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)  # warmup



def load_model(weights, device):
    model = attempt_load(weights, device=device, inplace=True)
    stride = max(int(model.stride.max()), 32)  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    model.float()
    return model  # explicitly assign for to(), cpu(), cuda(), half()


def main(quant_mode, model, imgsz):
    imgsz = (1, 1, imgsz, imgsz)
    device=torch.device('cpu')
    # weights_path="../01_trained_model/runs/train/exp13/weights/best.pt"

    # 加载yolo模型
    # model = load_model(weights_path, device)
    im = torch.randn(imgsz, dtype=torch.float, device=device)  # input
    model(im)  # warmup

    quantizer = torch_quantizer(quant_mode, model, (im), device=device)

    quant_model = quantizer.quant_model
    quant_model = quant_model.to(device)
    quant_model(im)

    # import val as validate  # for end-of-epoch mAP
    
    # results, maps, _ = validate.run(data_dict,
    #                             batch_size=batch_size // WORLD_SIZE * 2,
    #                             imgsz=imgsz,
    #                             half=amp,
    #                             model=ema.ema,
    #                             single_cls=single_cls,
    #                             dataloader=val_loader,
    #                             save_dir=save_dir,
    #                             plots=False,
    #                             callbacks=callbacks,
    #                             compute_loss=compute_loss)

    if quant_mode == 'calib':
        quantizer.export_quant_config()

    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False)
        # quantizer.export_onnx_model()


if __name__ == '__main__':

    device = torch.device('cpu')

    model = Net()
    state_dict = torch.load('ckpt/mnist.pth')
    model.load_state_dict(state_dict)

    # model.quant()

    model.to(device)

    # 在测试集上评估模型
    model.eval()

    # weights_path = '/weight.pt'

    imgsz = 28
    # print('imgsz:', imgsz)

    main(quant_mode='calib', model=model, imgsz=imgsz)
    main(quant_mode='test', model=model, imgsz=imgsz)

