from GANLatentDiscovery.loading import load_from_dir
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from inriaDataset import inriaDataset
from PyTorch_YOLOv3.pytorchyolo import detect, models
from torchvision import transforms
import os
import random
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
from adversarialYolo.load_data import PatchTransformer, PatchApplier

Seed = 1058
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)

imgSize = 416
batch_size = 8
t = 100
max_epoch = 1000
a = 0.01

experiment = "WithNoTransT50tiny"

image_dir = f"images/{experiment}"
lr = 0.005
rotate = False
noise = False
crease = False
if not os.path.exists(image_dir):
    os.makedirs(image_dir)



def combine(img, patch, label):
    # Combine the patch to the image
    if (label[0] == 0):
        label = (label * imgSize).int()
        x = label[1]
        y = label[2]
        w = int(label[3] * 0.7)
        h = int(label[4] * 0.7)
        y = y - h//5
        if w < h:
            h = w
        else:
            w = h
        resize = transforms.Resize((w, h))
        patch = resize(patch)
        img[:, int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = patch
    return img


def smoothness(patch):
    tvcomp1 = torch.sum(torch.abs(patch[:, :, 1:] - patch[:, :, :-1]+0.000001),0)
    tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
    tvcomp2 = torch.sum(torch.abs(patch[:, 1:, :] - patch[:, :-1, :]+0.000001),0)
    tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
    tv = tvcomp1 + tvcomp2
    return tv/torch.numel(patch)


_, G, _ = load_from_dir(
        './GANLatentDiscovery/models/pretrained/deformators/BigGAN/',
        G_weights='./GANLatentDiscovery/models/pretrained/generators/BigGAN/G_ema.pth')
G.set_classes(259)
len_z = G.dim_z
# z = torch.rand([1, len_z], device=torch.device('cuda'), requires_grad=True)
z = torch.normal(0.0, torch.ones(len_z)).to('cuda').unsqueeze(0).requires_grad_(True)
z.data = torch.round(z.data * 10000) * (10**-4)
test = G(z)
print(test.shape)
plt.imshow(test[0].cpu().detach().numpy().transpose(1,2,0))
path = os.path.join(image_dir, f"test.png")
plt.savefig(path)

writer = SummaryWriter(log_dir=f"natAdv_log/{experiment}", filename_suffix=experiment)

dataset = inriaDataset("dataset/inria/Train/pos", "dataset/inria/Train/pos/yolo-labels_yolov3", imgSize, 15)
train_size = int(len(dataset))
train = torch.utils.data.Subset(dataset, list(range(train_size)))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)

yolo = models.load_model("PyTorch_YOLOv3/config/yolov3-tiny.cfg", "PyTorch_YOLOv3/weights/yolov3-tiny.weights")

optimizer = torch.optim.Adam([z], lr=lr, betas=(0.5, 0.999))

metric = MeanAveragePrecision()
patch_transformer = PatchTransformer().cuda()
patch_applier = PatchApplier().cuda()

for epoch in range(max_epoch):
    print(f"Epoch {epoch}")
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()
        initialBoxes = detect.detect_image(yolo, images, conf_thres=0.5, classes=0)
        # initialProb = torch.mean(torch.max(initialBoxes[:,:,4], 1).values)
        # print(f"Probability before patching: {initialProb}")
        gt = []
        preds = []
        for i in range(images.shape[0]):
            currentBox = initialBoxes[i]
            gt.append(dict(boxes=currentBox[:, :4], labels=torch.zeros(currentBox.shape[0])))
        # Create a patch
        z.data = torch.clamp(z.data, min=-t, max=t)
        patch = G(z)
        patch = patch.squeeze(0).cuda()
        # Compute L_tv
        L_tv = smoothness(patch)
        L_det = torch.tensor([0], requires_grad=True, dtype=float).to("cuda")
        # for i in range(images.shape[0]):
        #     # Apply patch to each image in the batch
        #     label = labels[i]
        #     for box in label:
        #         images[i] = combine(images[i], patch, box)

        adv_batch, patch_set, _ = patch_transformer(adv_patch=patch, lab_batch=labels, img_size=imgSize, rand_loc=False, enable_blurred=False, with_crease=crease, do_rotate=rotate, enable_no_random=(not noise))
        images = patch_applier(images, adv_batch)
        boxes = detect.detect_image(yolo, images, conf_thres=0, classes=0)
        L_det = torch.tensor(0)
        for i in range(images.shape[0]):
            currentBox = boxes[i]
            max_prob = torch.max(currentBox[:, 4], 0).values
            L_det = L_det + max_prob
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
        print("gt")
        print(len(gt))
        print("preds")
        print(len(preds))

        L_det = L_det / images.shape[0]
        print(f"Detecton loss: {L_det}")
        L_tot = L_det + a*L_tv
        # L_tot = L_det
        print(f"Total loss: {L_tot}")
        torch.autograd.set_detect_anomaly(True)
        # print(z.grad)
        optimizer.zero_grad()
        L_tot.backward()
        optimizer.step()
    if epoch % 10 == 0:
        plt.imshow(patch.cpu().detach().numpy().transpose(1,2,0))
        patch_path = os.path.join(image_dir, "patch")
        if not os.path.exists(patch_path):
            os.makedirs(patch_path)
        plt.savefig(os.path.join(patch_path, f"{epoch}.png"))
        plt.imshow(images[0].cpu().detach().numpy().transpose(1,2,0))
        combine_path = os.path.join(image_dir, "combine")
        if not os.path.exists(combine_path):
            os.makedirs(combine_path)
        plt.savefig(os.path.join(combine_path, f"{epoch}.png"))
    mAP = metric.compute()
    writer.add_scalar("mAP", mAP["map"], global_step=epoch)
    print("mAP: ", mAP["map"])
    metric.reset()
