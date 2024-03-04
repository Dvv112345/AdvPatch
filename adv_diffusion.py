from diffusers import DDIMScheduler, UNet2DModel
import torch
import numpy as np
import cv2
from torchvision import transforms
import os
import random
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import argparse
import json

from inriaDataset import inriaDataset
from PyTorch_YOLOv3.pytorchyolo import detect, models
from advArt_util import smoothness, detect_loss, combine, perspective, wrinkles, rotate, noise, blur, getMask, saveImage
from pytorchYOLOv4.demo import DetectorYolov4
from yolov7 import custom_detector


def generateImage(start, scheduler, model):
    # run inference
    # Initial noise can be positive or negative.
    current = start
    for t in scheduler.timesteps:
        # print(t)
        noisy_residual = model(current, t).sample
        previous_noisy_sample = scheduler.step(noisy_residual, t, current).prev_sample
        current = previous_noisy_sample

    # save image
    image = (current / 2 + 0.5).clamp(0, 1).squeeze()
    return image



def trainPatch(args):
    img_size = args["imgSize"]
    batch_size = args["batch"]
    max_epoch = args["epoch"]
    a = args["a"]
    b = args["b"]
    lr = args["lr"]
    experiment = args["exp"]
    image_dir = f"advDiffusion/{experiment}"
    tiny = args["tiny"]
    model = args["model"]
    target_cls = args["targetClass"]
    patchSize = args["patchSize"]
    datasetPath = args["dataset"]
    labelPath = args["label"]
    minBox = args["imageFilter"]
    pretrainedDiffusion = args["diffusionModel"]
    limit = args["latentLimit"]

    # Set up directory for storing the result
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    else:
        os.system(f'rm -rf {image_dir}/*')

    with open(os.path.join(image_dir, "setting.json"), 'w') as f:
        json.dump(args, f, indent = 6, skipkeys=True, ensure_ascii=False, allow_nan=True)

    patch_path = os.path.join(image_dir, "patch")
    if not os.path.exists(patch_path):
        os.makedirs(patch_path)
    combine_path = os.path.join(image_dir, "combine")
    if not os.path.exists(combine_path):
        os.makedirs(combine_path)

    # load diffusion model and scheduler
    diffusionScheduler = DDIMScheduler.from_pretrained(pretrainedDiffusion)
    try:
        diffusionModel = UNet2DModel.from_pretrained(pretrainedDiffusion, use_safetensors=True).to("cuda")
    except OSError:
        diffusionModel = UNet2DModel.from_pretrained(pretrainedDiffusion).to("cuda")
    diffusionModel.eval()
    diffusionScheduler.set_timesteps(10)
    sample_size = diffusionModel.config.sample_size
    latent = torch.randn((1,3,sample_size, sample_size), device="cuda")
    initialLatent = latent.clone()
    latent.requires_grad_()

    initialPatch = generateImage(latent, diffusionScheduler, diffusionModel)
    saveImage(initialPatch, os.path.join(image_dir, "initial.png"))

    # Set up tensorboard
    writer = SummaryWriter(log_dir=f"advDiffusion_log/{experiment}", filename_suffix=experiment)

    # Load the dataset for training
    dataset = inriaDataset(datasetPath, labelPath, img_size, 14, minBox=minBox)
    train_size = int(len(dataset))
    print("Size of dataset: ", len(dataset))
    dataset.filter()
    if train_size > len(dataset):
        train_size = len(dataset)
    train = torch.utils.data.Subset(dataset, list(range(train_size)))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=2)

     # Load the image detection model
    if model == "v3":
        print("Using YOLOv3")
        if tiny:
            yolo = models.load_model("PyTorch_YOLOv3/config/yolov3-tiny.cfg", "PyTorch_YOLOv3/weights/yolov3-tiny.weights")
        else:
            yolo = models.load_model("PyTorch_YOLOv3/config/yolov3.cfg", "PyTorch_YOLOv3/weights/yolov3.weights")
    # elif model == "v4":
    #     print("Using YOLOv4")
    #     detector = DetectorYolov4(show_detail=False, tiny=tiny)
    elif model == "v7":
        print("Using YOLOv7")
        detector = custom_detector.Detector("yolov7/yolov7.pt")
    elif model == "faster":
        print("Using Faster-RCNN")
        detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").cuda()
        detector.eval()

    counter = 0
    metric = MeanAveragePrecision()
    optimizer = torch.optim.Adam([latent], lr=lr, amsgrad=True)

    for epoch in range(max_epoch):
        print(f"Epoch {epoch}")
        for images, labels in train_loader:
            #     optimizer = torch.optim.Adam([patch_temp], lr=lr, amsgrad=True)
            images = images.cuda()
            # labels = labels.cuda()
            if model == "v3":
                initialBoxes = detect.detect_image(yolo, images, conf_thres=0.5, classes=0)
                # print(initialBoxes[0].shape)
                # print(initialBoxes[0])
                # print(initialBoxes)
            elif model == "v4":
                _, _, initialBoxes = detector.detect(input_imgs=images, cls_id_attacked=0, clear_imgs=None, with_bbox=True, conf_thresh=0.5) 
                # print(initialBoxes[0].shape)
                # print(initialBoxes[0])
                # print(initialBoxes)
            elif model == "v7":
                detector = custom_detector.Detector("yolov7/yolov7.pt")
                initialBoxes = detector.detect(images, conf_thres=0.5, classes=0)
                # print(initialBoxes[0].shape)
                # print(initialBoxes[0])
            elif model == "faster":
                initialBoxes = detector(images)
                for i in range(len(initialBoxes)):
                    initialBoxes[i] = torch.cat([initialBoxes[i]["boxes"], initialBoxes[i]["scores"].unsqueeze(1), initialBoxes[i]["labels"].unsqueeze(1)], dim=1)
                    initialBoxes[i] = initialBoxes[i][initialBoxes[i][:,5] == 1]
                    initialBoxes[i] = initialBoxes[i][initialBoxes[i][:,4] >= 0.5]
                    # path = os.path.join(image_dir, f"detail_{i}.png")
                    # Image.fromarray((images[i].cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(path)
            # initialProb = torch.mean(torch.max(initialBoxes[:,:,4], 1).values)
            # print(f"Initial Probability: {initialProb}")
            # print(initialBoxes[0])
            gt = []
            preds = []
            labels = []
            for i in range(images.shape[0]):
                currentBox = initialBoxes[i].cuda()
                # print(currentBox)
                gt.append(dict(boxes=currentBox[:, :4],
                labels=torch.zeros(currentBox.shape[0])))
                # if i == 0:
                #     print("Initial:")
                #     print(currentBox)
                if model == "v3" or "v7" or "faster":
                    currentBox = currentBox / img_size
                # Convert (x1,y1,x2,y2) to (x,y,w,h)
                width = currentBox[:14,2] - currentBox[:14, 0]
                width = torch.where((width == 0), 1, width)
                height = currentBox[:14,3] - currentBox[:14, 1]
                height = torch.where((height==0), 1, height)
                center_x = currentBox[:14, 0] + width/2
                center_y = currentBox[:14, 1] + height/2
                label = torch.cat((currentBox[:14, 5].unsqueeze(1), center_x.unsqueeze(1), center_y.unsqueeze(1), width.unsqueeze(1), height.unsqueeze(1)), 1).cuda()
                # print(label.shape)
                labels.append(label)

            # print(labels[0])

            # Generate the patch and Compute L_tv
            patch = generateImage(latent, diffusionScheduler, diffusionModel)
            L_tv = smoothness(patch)
            
            advImages = torch.zeros(images.shape).cuda()
            patch_o = patch
            if args["blur"]:
                patch_o = blur(patch_o)
            for i in range(len(labels)):
                patch_t = patch_o
                trans_prob = torch.rand([1])
                # trans_prob = 1
                if args["noise"]:
                    patch_t = noise(patch_t)
                if trans_prob > 0.6:
                    if args["rotate"]:
                        patch_t = rotate(patch_t, (sample_size, sample_size))
                    if args["persp"]:
                        patch_t = perspective(patch_t)
                    if args["wrinkle"]:
                        patch_t = wrinkles(patch_t)

                patch_batch, mask = getMask(patch_t, labels[i].unsqueeze(0), img_size, patchSize)
                advImages[i] = combine(images[i], patch_batch, mask)

            if args["saveDetail"]:
                path = os.path.join(patch_path, f"detail_epoch{epoch}_{counter}.png")
                Image.fromarray((patch_t.cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(path)
                path = os.path.join(combine_path, f"detail_epoch{epoch}_{counter}.png")
                Image.fromarray((advImages[0].cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(path)
            
            boxes = []
            if model == "v3":
                boxes = detect.detect_image(yolo, advImages, conf_thres=0, classes=0, target=target_cls)
            # elif model == "v4":
            #     _, _, boxes = detector.detect(input_imgs=advImages, cls_id_attacked=0, clear_imgs=None, with_bbox=True, conf_thresh=0)
            elif model == "v7":
                boxes = detector.detect(advImages, conf_thres=0, classes=0)
            elif model == "faster":
                boxes = detector(advImages)
                for i in range(len(boxes)):
                    boxes[i] = torch.cat([boxes[i]["boxes"], boxes[i]["scores"].unsqueeze(1), boxes[i]["labels"].unsqueeze(1)], dim=1)
                    boxes[i] = boxes[i][boxes[i][:,5] == 1]
            # print(boxes[0])
            prob = []
            maxProb = torch.zeros(images.shape[0])
            for i in range(images.shape[0]):
                # print(i)
                if target_cls is None:
                    if boxes[i][:, 4].shape[0] == 0:
                        prob.append(torch.tensor([0]).float())
                    else:
                        prob.append(boxes[i][:,4])
                else:
                    prob.append(boxes[i][:,6])
                if boxes[i][:, 4].shape[0] == 0:
                    maxProb[i] = torch.tensor(0).float()
                else:
                    maxProb[i] = torch.max(boxes[i][:,4])
                # print(maxProb[i])
            # print(boxes.shape)
            max_prob = torch.mean(maxProb).cuda()
            # max_prob_obj_cls, overlap_score, boxes = detector.detect(input_imgs=advImages, cls_id_attacked=0, clear_imgs=None, with_bbox=True)
            # max_prob = torch.mean(max_prob_obj_cls)
            L_det = 0
            L_det = detect_loss(prob, labels).cuda()
            # print(L_det)
            # L_det = max_prob
            for i in range(images.shape[0]):
                currentBox = boxes[i]
                if len(currentBox.shape) == 2:
                    currentBox = currentBox[currentBox[:,4]>0.5]
                    # if i == 0:
                    #     print("Final:")
                    #     print(currentBox)
                    preds.append(dict(boxes=currentBox[:, :4],
                    scores=currentBox[:, 4],
                    labels=torch.zeros(currentBox.shape[0])))
                else:
                    preds.append(dict(boxes=torch.tensor([]),
                    scores=torch.tensor([]),
                    labels=torch.tensor([])))
            metric.update(preds, gt)
            # print("Preds:")
            # print(preds[0])
            # print("GT:")
            # print(gt[0])
            # mAP = metric.compute()
            # print("mAP: ", mAP["map"])

            # Print the loss
            print(f"Detecton loss: {L_det}")
        
            L_tot = a*L_det + b*L_tv
            # L_tot = L_det
            writer.add_scalar("Detection_Prob", max_prob, global_step=counter)
            lossTag = {"L_tot": L_tot, "L_det": L_det, "L_tv": L_tv}
            writer.add_scalars("Loss", lossTag, counter)
            # torch.autograd.set_detect_anomaly(True)

            # filter = torch.zeros(3, 3, 3, 3).cuda()
            # avg_filter = torch.ones(3,3) / 9
            # for i in range(3):
            #     filter[i,i] = avg_filter
            # print(filter)

            L_tot.backward()
            # patch.grad = F.conv2d(patch.grad, filter, padding="same")
            optimizer.step()
            # print(f"Latent gradient: ", latent.grad)
            print(f"Latent gradient sum of absolute: {torch.sum(torch.abs(latent.grad))}")
            latent.data = torch.clamp(latent.data, min=-limit, max=limit)
            print(f"Sum of change: {torch.sum(torch.abs(latent.data - initialLatent.data))}")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            counter += 1
        print(f"End of epoch {epoch}")
        mAP = metric.compute()
        writer.add_scalar("mAP", mAP["map"], global_step=epoch)
        print("mAP: ", mAP["map"])
        metric.reset()
        if not args["saveDetail"] and (epoch % 10 == 0 or epoch == max_epoch - 1):
            Image.fromarray((patch.cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(os.path.join(patch_path, f"{epoch}.png"))
            Image.fromarray((advImages[0].cpu().detach().numpy().transpose(1,2,0)* 255).astype(np.uint8)).save(os.path.join(combine_path, f"{epoch}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=1, type=float)
    parser.add_argument("--b", default=0.5, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=10000, type=int)
    parser.add_argument("--exp", default="test")
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--targetClass", default=None, type=int)
    parser.add_argument("--dataset", default="dataset/inria/Train/pos")
    parser.add_argument("--label", default="dataset/inria/Train/pos/yolo-labels_yolov4")
    parser.add_argument("--model", default="v3")
    parser.add_argument("--diffusionModel", default = "google/ddpm-ema-church-256")
    parser.add_argument("--tiny", action='store_true')
    parser.add_argument("--saveDetail", action='store_true')
    parser.add_argument("--noise", action='store_true')
    parser.add_argument("--rotate", action='store_true')
    parser.add_argument("--blur", action='store_true')
    parser.add_argument("--persp", action='store_true')
    parser.add_argument("--wrinkle", action='store_true')
    parser.add_argument("--patchSize", default=0.6, type=float)
    parser.add_argument("--imgSize", default = 416, type=int)
    parser.add_argument("--imageFilter", default=0, type=float)
    parser.add_argument("--latentLimit", default=0, type=int)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    trainPatch(vars(args))