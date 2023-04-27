from inriaDataset import inriaDataset
from advArt_util import combine
import argparse
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--batch", default=8, type=int)
parser.add_argument("--patchSize", default=0.5, type=float)
parser.add_argument("--imageFilter", default=0, type=float)
parser.add_argument("--name", default="test")
parser.add_argument("--resize", default=400, type=int)
parser.add_argument("--patch", default="target_art.jpeg")
parser.add_argument("--dataset", default="dataset/inria/Train/pos")
parser.add_argument("--label", default="dataset/inria/Train/pos/yolo-labels_yolov4")
args = parser.parse_args()

targetResize = args.resize
img_size = 416

patch = Image.open(args.patch).convert('RGB')
transform = transforms.ToTensor()
patch = transform(patch)
patch = patch.cuda()
print("Initial size of patch image: ", patch.shape)
if targetResize > 0:
    resize = transforms.Resize((targetResize, targetResize))
    patch = resize(patch)
print(patch.shape)

path = os.path.join("combine", args.name)
if not os.path.exists(path):
        os.makedirs(path)

def getMask(patch, labels):
    if patch.size(-1) > img_size:
        resize = transforms.Resize((img_size, img_size))
        patch = resize(patch)
    # Make a batch of patch
    patch_batch = patch.expand(labels.size(0), labels.size(1), -1, -1, -1)
    # print(patch_batch.shape)

    # Create mask
    clsId = torch.narrow(labels, 2, 0, 1)
    # print(clsId.shape)
    clsId.data = torch.clamp(clsId.data, min=0, max=1)
    clsMask = clsId.unsqueeze(-1)
    clsMask = clsMask.unsqueeze(-1)
    clsMask = clsMask.expand(-1, -1, patch_batch.size(-3), patch_batch.size(-2), patch_batch.size(-1))
    # print(clsMask.shape)
    mask = torch.cuda.FloatTensor(clsMask.size()).fill_(1) - clsMask
    # print(mask.shape)
    padW = (img_size - mask.size(-1)) / 2
    padH = (img_size - mask.size(-2)) / 2
    mypad = torch.nn.ConstantPad2d((int(padW), int(padW), int(padH), int(padH)), 0)
    patch_batch = mypad(patch_batch)
    mask = mypad(mask)
    # print(mask.shape)

    flattenSize = labels.size(0) * labels.size(1)

    lab_scaled = labels * img_size
    patch_size = args.patchSize
    smallSide = torch.where(lab_scaled[:,:,3] < lab_scaled[:,:, 4], lab_scaled[:,:,3], lab_scaled[:,:, 4])
    target_size = smallSide.mul(patch_size)
    # target_size = torch.sqrt(((lab_scaled[:, :, 3].mul(patch_size)) ** 2) + ((lab_scaled[:, :, 4].mul(patch_size)) ** 2))
    target_x = labels[:, :, 1].view(flattenSize)
    target_y = (labels[:, :, 2]- 0.1*labels[:, :, 4]).view(flattenSize)
    tx = (1-2*target_x)
    ty = (1-2*target_y)

    # print(target_size.shape)
    scale = patch.size(-1)/target_size
    scale = scale.view(flattenSize)
    theta = torch.cuda.FloatTensor(flattenSize, 2, 3).fill_(0)
    theta[:, 0, 0] = scale
    theta[:, 0, 2] = tx * scale
    theta[:, 1, 1] = scale
    theta[:, 1, 2] = ty * scale
    maskShape = mask.shape
    patch_batch = patch_batch.view(flattenSize, maskShape[2], maskShape[3], maskShape[4])
    mask = mask.view(flattenSize, maskShape[2], maskShape[3], maskShape[4])
    grid = F.affine_grid(theta, patch_batch.shape, align_corners=False)
    patch_t = F.grid_sample(patch_batch, grid, align_corners=False)
    mask_t = F.grid_sample(mask, grid, align_corners=False)
    patch_t = patch_t.view(maskShape)
    mask_t = mask_t.view(maskShape)
    return patch_t, mask_t



# dataset = inriaDataset("dataset/inria/Train/pos", "dataset/inria/Train/pos/yolo-labels_yolov4", img_size, 14, minBox=args.imageFilter)

dataset = inriaDataset(args.dataset, args.label, img_size, 14, minBox=args.imageFilter)

dataset.filter()


for i in range(args.batch):
    image, label = dataset[i]
    label = label.cuda()
    image = image.cuda()
    patch_batch, mask = getMask(patch, label.unsqueeze(0))
    result = combine(image, patch_batch, mask).squeeze()
    Image.fromarray((result.cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(os.path.join(path, f"combined_{i}.png"))
    Image.fromarray((image.cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(os.path.join(path, f"original_{i}.png"))
