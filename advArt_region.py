import torch
import numpy as np
import cv2
from torchvision import transforms
import os
import random
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import argparse
from math import pi

from inriaDataset import inriaDataset
from PyTorch_YOLOv3.pytorchyolo import detect, models
from advArt_util import smoothness, similiar, detect_loss, combine, perspective, wrinkles, rotate, noise, NPS
# from adversarialYolo.load_data import PatchTransformer, PatchApplier, InriaDataset 
# from PyTorchYOLOv3.detect import DetectorYolov3
# from pytorchYOLOv4.demo import DetectorYolov4

parser = argparse.ArgumentParser()
parser.add_argument("--a", default=1, type=float)
parser.add_argument("--b", default=0.5, type=float)
parser.add_argument("--c", default=1, type=float)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--epoch", default=2000, type=int)
parser.add_argument("--allTrans", action='store_true')
parser.add_argument("--exp", default="test")
parser.add_argument("--resize", default=400, type=int)
parser.add_argument("--batch", default=8, type=int)
parser.add_argument("--targetClass", default=None, type=int)
parser.add_argument("--target", default="target_art.jpeg")
parser.add_argument("--saveTrans", action='store_true')
parser.add_argument("--noise", action='store_true')
parser.add_argument("--rotate", action='store_true')
parser.add_argument("--persp", action='store_true')
parser.add_argument("--wrinkle", action='store_true')
parser.add_argument("--patchSize", default=0.5, type=float)
parser.add_argument("--imageFilter", default=0, type=float)
parser.add_argument("--note", default="")
args = parser.parse_args()

# Set the hyperparameters
img_size = 416
batch_size = args.batch
max_epoch = args.epoch
a = args.a
b = args.b
c = args.c
lr = args.lr
target_path = args.target
experiment = args.exp
image_dir = f"artImg/{experiment}"
targetResize = args.resize
do_transform = args.allTrans
max_patch = 0
min_patch = img_size
target_cls = args.targetClass
regionX1 = 0
regionX2 = 100
regionY1 = 0
regionY2 = 400

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

with open(os.path.join(image_dir, "setting.txt"), 'w') as f:
    f.write(f"Optimize on a region of the patch\nImage size: {img_size}\n{args}")


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
    batch_max = torch.max(target_size).detach().cpu().item()
    batch_min = torch.min(target_size).detach().cpu().item()
    global max_patch
    global min_patch
    if batch_max > max_patch:
        max_patch = batch_max
    if batch_min < min_patch:
        min_patch = batch_min
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


# Get the target 
target = Image.open(target_path).convert('RGB')
transform = transforms.ToTensor()
target = transform(target)
target = target.cuda()
print("Initial size of target image: ", target.shape)
if targetResize > 0:
    resize = transforms.Resize((targetResize, targetResize))
    target = resize(target)
print(target.shape)
patch = target.detach()
region = patch[:, regionY1:regionY2, regionX1:regionX2].clone()
region.requires_grad_(True)
patch[:, regionY1:regionY2, regionX1:regionX2] = 0

path = os.path.join(image_dir, f"region.png")
Image.fromarray((patch.cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(path)

patch[:, regionY1:regionY2, regionX1:regionX2] = region
path = os.path.join(image_dir, f"initial.png")
Image.fromarray((patch.cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(path)

# Set tensorboard
writer = SummaryWriter(log_dir=f"advArt_log/{experiment}", filename_suffix=experiment)

# Load the dataset for training
dataset = inriaDataset("dataset/inria/Train/pos", "dataset/inria/Train/pos/yolo-labels_yolov4", img_size, 14, minBox=args.imageFilter)
# dataset = InriaDataset("dataset/inria/Train/pos", "dataset/inria/Train/pos/yolo-labels_yolov3", max_lab=14, imgsize=img_size)
train_size = int(len(dataset) * 0.2)
print("Size of dataset: ", len(dataset))
dataset.filter()
if train_size > len(dataset):
    train_size = len(dataset)
train = torch.utils.data.Subset(dataset, list(range(train_size)))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=2)

# Load the image detection model
yolo = models.load_model("PyTorch_YOLOv3/config/yolov3-custom.cfg", "PyTorch_YOLOv3/weights/yolov3.weights")

# Set the optimizer
# optimizer = torch.optim.SGD([patch], lr=lr, momentum=0.9)
optimizer = torch.optim.Adam([region], lr=lr, amsgrad=True)
# patch_transformer = PatchTransformer().cuda()
# patch_applier = PatchApplier().cuda()

counter = 0
metric = MeanAveragePrecision()


for epoch in range(max_epoch):
    print(f"Epoch {epoch}")
    for images, labels in train_loader:
        images = images.cuda()
        # labels = labels.cuda()
        initialBoxes = detect.detect_image(yolo, images, conf_thres=0, classes=0).cuda()
        # initialProb = torch.mean(torch.max(initialBoxes[:,:,4], 1).values)
        # print(f"Initial Probability: {initialProb}")
        # _, _, initialBoxes = detector.detect(input_imgs=images, cls_id_attacked=0, clear_imgs=None, with_bbox=True)
        gt = []
        preds = []
        for i in range(images.shape[0]):
            currentBox = initialBoxes[i]
            if len(currentBox.shape) == 2:
                currentBox[currentBox[:,4]<=0.5] = img_size
                initialBoxes[i] = currentBox
                currentBox = currentBox[currentBox[:,4] < img_size]
                gt.append(dict(boxes=currentBox[:, :4],
                labels=torch.zeros(currentBox.shape[0])))
            else:
                gt.append(dict(boxes=torch.tensor([]),
                labels=torch.tensor([])))

        # print(labels[0])
        initialBoxes = initialBoxes / img_size
        width = initialBoxes[:,:14,2] - initialBoxes[:, :14, 0]
        width = torch.where((width == 0), 1, width)
        height = initialBoxes[:,:14,3] - initialBoxes[:, :14, 1]
        height = torch.where((height==0), 1, height)
        center_x = initialBoxes[:, :14, 0] + width/2
        center_y = initialBoxes[:, :14, 1] + height/2
        labels = torch.cat((initialBoxes[:, :14, 5].unsqueeze(2), center_x.unsqueeze(2), center_y.unsqueeze(2), width.unsqueeze(2), height.unsqueeze(2)), 2).cuda()
        # print(labels[0])

        # Compute L_tv and L_sim
        L_tv = smoothness(patch)
        L_sim = similiar(patch, target)
    
        # patch_t = patch
        # trans_prob = torch.rand([1])
        # # trans_prob = 1
        # if do_transform or args.noise:
        #     patch_t = noise(patch_t)
        # if trans_prob > 0.6:
        #     if do_transform or args.rotate:
        #         patch_t = rotate(patch_t)
        #     if args.persp:
        #         patch_t = perspective(patch_t)
        #     if do_transform or args.wrinkle:
        #         patch_t = wrinkles(patch_t)
    
        #     # adv_batch, patch_set, _ = patch_transformer(adv_patch=patch_t, lab_batch=labels, img_size=img_size, rand_loc=False, enable_blurred=False)
        #     # advImages = patch_applier(images, adv_batch)
        # patch_batch, mask = getMask(patch_t, labels)
        # advImages = combine(images, patch_batch, mask)
        
        advImages = torch.zeros(images.shape).cuda()
        for i in range(labels.size(0)):
            patch_t = patch
            trans_prob = torch.rand([1])
            # trans_prob = 1
            if do_transform or args.noise:
                patch_t = noise(patch_t)
            if trans_prob > 0.6:
                if do_transform or args.rotate:
                    patch_t = rotate(patch_t, targetResize)
                if args.persp:
                    patch_t = perspective(patch_t)
                if do_transform or args.wrinkle:
                    patch_t = wrinkles(patch_t)
        
                # adv_batch, patch_set, _ = patch_transformer(adv_patch=patch_t, lab_batch=labels, img_size=img_size, rand_loc=False, enable_blurred=False)
                # advImages = patch_applier(images, adv_batch)
            patch_batch, mask = getMask(patch_t, labels[i].unsqueeze(0))
            advImages[i] = combine(images[i], patch_batch, mask)

        # path = os.path.join(image_dir, f"detail.png")
        # Image.fromarray((patch_t.cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(path)
        # path = os.path.join(image_dir, f"detailCombine.png")
        # Image.fromarray((advImages[0].cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(path)


        boxes = detect.detect_image(yolo, advImages, conf_thres=0, classes=0, target=target_cls)
        # print(boxes.shape)
        max_prob = torch.mean(torch.max(boxes[:,:,4], 1).values).cuda()
        # max_prob_obj_cls, overlap_score, boxes = detector.detect(input_imgs=advImages, cls_id_attacked=0, clear_imgs=None, with_bbox=True)
        # max_prob = torch.mean(max_prob_obj_cls)
        if target_cls is None:
            L_det = detect_loss(boxes[:,:,4], labels).cuda()
        else:
            L_det = detect_loss(boxes[:,:,6], labels).cuda()
        # L_det = max_prob
        for i in range(images.shape[0]):
            currentBox = boxes[i]
            if len(currentBox.shape) == 2:
                currentBox = currentBox[currentBox[:,4]>0.5]
                preds.append(dict(boxes=currentBox[:, :4],
                scores=currentBox[:, 4],
                # scores=currentBox[:, 4]*currentBox[:, 5],
                labels=torch.zeros(currentBox.shape[0])))
            else:
                preds.append(dict(boxes=torch.tensor([]),
                scores=torch.tensor([]),
                labels=torch.tensor([])))
        metric.update(preds, gt)

        # Print the loss
        print(f"Detecton loss: {L_det}")
        # print(f"Detecton loss gradient: {L_det.grad_fn}")
        # print(f"Similarity loss: {L_sim}")
        # print(f"Probability check: {probCheck}")
    
        L_tot = a*L_det + b*L_tv + c*L_sim
        # L_tot = L_det
        # probTag = {"Initial_Prob": initialProb, "Final_Prob": L_det}
        writer.add_scalar("Detection_Prob", max_prob, global_step=counter)
        lossTag = {"L_tot": L_tot, "L_det": L_det, "L_sim": L_sim, "L_tv": L_tv}
        writer.add_scalars("Loss", lossTag, counter)
        torch.autograd.set_detect_anomaly(True)

        # filter = torch.zeros(3, 3, 3, 3).cuda()
        # avg_filter = torch.ones(3,3) / 9
        # for i in range(3):
        #     filter[i,i] = avg_filter
        # print(filter)

        L_tot.backward()
        # patch.grad = F.conv2d(patch.grad, filter, padding="same")
        # print(f"Patch gradient: {patch.grad}")
        optimizer.step()
        patch = patch.detach()
        patch[:, regionY1:regionY2, regionX1:regionX2] = region
        # patch.data = F.conv2d(patch.data, filter, padding="same")
        # print(patch.shape)
        patch.data = torch.clamp(patch.data, min=0, max=1)
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        counter += 1
    print(f"End of epoch {epoch}")
    print(f"Max size of patch: {max_patch}; Min size of patch: {min_patch}")
    mAP = metric.compute()
    writer.add_scalar("mAP", mAP["map_50"], global_step=epoch)
    print("mAP: ", mAP["map_50"])
    metric.reset()
    if epoch % 10 == 0:
            patch_path = os.path.join(image_dir, "patch")
            if not os.path.exists(patch_path):
                os.makedirs(patch_path)
            Image.fromarray((patch.cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(os.path.join(patch_path, f"{epoch}.png"))
            combine_path = os.path.join(image_dir, "combine")
            if not os.path.exists(combine_path):
                os.makedirs(combine_path)
            Image.fromarray((advImages[0].cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(os.path.join(combine_path, f"{epoch}.png"))
            if args.saveTrans:
                trans_path = os.path.join(image_dir, "transformation")
                if not os.path.exists(trans_path):
                    os.makedirs(trans_path)
                Image.fromarray((patch_t.cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(os.path.join(trans_path, f"{epoch}.png"))