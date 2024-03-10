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
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from tqdm.auto import tqdm
from torchvision import transforms
from math import sqrt
import gc
import sys

from inriaDataset import inriaDataset
from PyTorch_YOLOv3.pytorchyolo import detect, models
from advArt_util import smoothness, detect_loss, combine, perspective, wrinkles, rotate, noise, blur, getMask
from pytorchYOLOv4.demo import DetectorYolov4
from yolov7 import custom_detector


def getsize(obj, name):
    from types import ModuleType, FunctionType
    """sum size of object & members."""
    BLACKLIST = type, ModuleType, FunctionType
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
                if torch.is_tensor(obj) and obj.grad is not None:
                    print(name, "- Have grad")
                    del obj.grad
                    obj.requires_grad_(False)
                if torch.is_tensor(obj) and obj.grad_fn is not None:
                    print(name, "- Have grad fn")
        objects = gc.get_referents(*need_referents)
    return size

def get_var_size(space):
    sizes = []
    for name, value in space.items():
        size = getsize(value, name)
        sizes.append((name, size))
    sizes.sort(key=lambda elem: elem[1], reverse=True)
    return sizes

def diffuseLatent(latent, scheduler, t_start, height, width, in_channels):
    latents = latent * sqrt(scheduler.alphas_cumprod[t_start]) + torch.randn(
        (1, in_channels, height // 8, width // 8),
        device="cuda",) * sqrt(1-scheduler.alphas_cumprod[t_start])
    return latents

def decodeImage(latent, vae):
    latent = 1 / 0.18215 * latent
    image = vae.decode(latent).sample
    image = (image / 2 + 0.5)
    return image

def saveImage(image, path):
    image = image.clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image.save(path)

def finalLatent(start, scheduler, unet, text_embeddings, guidance_scale, use_grad=True):
    # run inference
    # Initial noise can be positive or negative.
    latents = start
    for t in tqdm(scheduler.timesteps):
        # print(torch.cuda.memory_summary())
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        if not use_grad:
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        else:
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    # decode image
    return latents


def encodeText(prompt, tokenizer, text_encoder):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * 1, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


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
    torch_device = "cuda"
    prompt = ["A high quality photo of a Pomeranian, clean background."]

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
    vae = AutoencoderKL.from_pretrained(pretrainedDiffusion, subfolder="vae", use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained(pretrainedDiffusion, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrainedDiffusion, subfolder="text_encoder", use_safetensors=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrainedDiffusion, subfolder="unet", use_safetensors=True
    )
    scheduler = DDIMScheduler.from_pretrained(pretrainedDiffusion, subfolder="scheduler")
    
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)
    unet.eval()
    vae.eval()
    text_encoder.eval()

    dif_height = 512  # default height of Stable Diffusion
    dif_width = 512  # default width of Stable Diffusion
    dif_guidance_scale = 7.5  # Scale for classifier-free guidance
    t_start = 250
    stepSize = 167
    inference_steps = list(range(t_start, 0, -stepSize))
    if inference_steps[-1] != 1:
        inference_steps.append(1)
    # print(inference_steps)
    num_inference_steps = len(inference_steps)  # Number of denoising steps

    # encode prompt
    text_embedding = encodeText(prompt, tokenizer, text_encoder).detach()
    text_embedding.requires_grad = False
    del tokenizer, text_encoder

    # Initialize latent
    generator = torch.Generator("cuda").manual_seed(123)
    initial_latents = torch.randn(
        (1, unet.config.in_channels, dif_height // 8, dif_width // 8),
        generator=generator,
        device=torch_device,
    )
    initial_latents = initial_latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps * 2)
    print("Latent initialized")

    # Initial image and latent
    latent = finalLatent(initial_latents, scheduler, unet, text_embedding, dif_guidance_scale, False)
    initialImage = decodeImage(latent, vae)
    saveImage(initialImage, os.path.join(image_dir, "initialImage.png"))
    latent.requires_grad_()
    print("Initial image generated")
    del initialImage
    initial_latents.to("cpu")

    # Reset scheduler timesteps
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = inference_steps

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
            detector = models.load_model("PyTorch_YOLOv3/config/yolov3-tiny.cfg", "PyTorch_YOLOv3/weights/yolov3-tiny.weights")
        else:
            detector = models.load_model("PyTorch_YOLOv3/config/yolov3.cfg", "PyTorch_YOLOv3/weights/yolov3.weights")
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

    # print(get_var_size(locals()))
    torch.cuda.empty_cache()
    for epoch in range(max_epoch):
        print(f"Epoch {epoch}")
        mAP_sum = 0
        # print(get_var_size(locals()))
        for images, labels in train_loader:
            metric = MeanAveragePrecision()
            unet = UNet2DConditionModel.from_pretrained(
                pretrainedDiffusion, subfolder="unet", use_safetensors=True
            ).to("cuda")
            # print("Memory at start")
            # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
            images = images.cuda()
            # labels = labels.cuda()
            if model == "v3":
                # if tiny:
                #     detector = models.load_model("PyTorch_YOLOv3/config/yolov3-tiny.cfg", "PyTorch_YOLOv3/weights/yolov3-tiny.weights")
                # else:
                #     detector = models.load_model("PyTorch_YOLOv3/config/yolov3.cfg", "PyTorch_YOLOv3/weights/yolov3.weights")
                initialBoxes = detect.detect_image(detector, images, conf_thres=0.5, classes=0)
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
            diffused_latent = diffuseLatent(latent, scheduler, t_start, dif_height, dif_width, unet.config.in_channels)
            new_latent = finalLatent(diffused_latent, scheduler, unet, text_embedding, dif_guidance_scale)
            # latent.requires_grad_()
            # prev_opt_state = optimizer.state_dict()
            # optimizer = torch.optim.Adam([latent], lr=lr, amsgrad=True)
            # optimizer.load_state_dict(prev_opt_state)
            patch = decodeImage(new_latent, vae)
            # print("Memory after decode image")
            # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
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
                        patch_t = rotate(patch_t, (dif_width, dif_height))
                    if args["persp"]:
                        patch_t = perspective(patch_t)
                    if args["wrinkle"]:
                        patch_t = wrinkles(patch_t)

                patch_batch, mask = getMask(patch_t, labels[i].unsqueeze(0), img_size, patchSize)
                advImages[i] = combine(images[i], patch_batch, mask)

            if args["saveDetail"]:
                path = os.path.join(patch_path, f"detail_epoch{epoch}_{counter}.png")
                saveImage(patch_t, path)
                path = os.path.join(combine_path, f"detail_epoch{epoch}_{counter}.png")
                saveImage(advImages[0], path)
            
            boxes = []
            if model == "v3":
                boxes = detect.detect_image(detector, advImages, conf_thres=0, classes=0, target=target_cls)
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
            # print("Memory after backward")
            # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
            # patch.grad = F.conv2d(patch.grad, filter, padding="same")
            # cur = latent.clone()
            optimizer.step()
            mAP_sum += metric.compute()["map"]
            # print("Difference after step", (latent - cur).sum())
            # print(f"Latent gradient: ", latent.grad)
            print(f"Latent gradient sum of absolute: {torch.sum(torch.abs(latent.grad))}")
            if limit > 0:
                latent.data = torch.clamp(latent.data, min=-limit, max=limit)
            print(f"Sum of change: {torch.sum(torch.abs(latent.data - initial_latents.data))}")
            optimizer.zero_grad()
            del initialBoxes, boxes, prob, maxProb, labels, max_prob, diffused_latent, new_latent, patch, L_tot, L_det, advImages, gt, preds, images, L_tv, lossTag, patch_batch, mask, width, height, center_x, center_y, label, patch_o, patch_t, unet, currentBox, metric
            latent.grad = None
            gc.collect()
            torch.cuda.empty_cache()
            # print("Memory after del")
            # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
            # print(get_var_size(locals()))
            counter += 1

        print(f"End of epoch {epoch}")
        mAP = mAP_sum / len(train_loader)
        writer.add_scalar("mAP", mAP, global_step=epoch)
        print("mAP: ", mAP)
        metric.reset()
        if not args["saveDetail"] and (epoch % 10 == 0 or epoch == max_epoch - 1):
            saveImage(patch, os.path.join(patch_path, f"{epoch}.png"))
            saveImage(advImages[0], os.path.join(combine_path, f"{epoch}.png"))

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
    parser.add_argument("--diffusionModel", default = "CompVis/stable-diffusion-v1-4")
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
