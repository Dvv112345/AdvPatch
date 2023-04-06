from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import fnmatch
import numpy as np
import torch

class inriaDataset(Dataset):
    def __init__(self, img_dir, lab_dir, imgSize, labSize, minBox = 0.05):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images= n_jpg_images + n_png_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.imgSize = imgSize
        self.labSize = labSize
        self.minBox = minBox
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index < len(self), "Index range error"
        img_path = self.img_paths[index]
        lab_path = self.lab_paths[index]
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])
        if label.ndim == 1:
            label = np.expand_dims(label, 0)
        label = torch.from_numpy(label).float()
        image, label = self.pad_image(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        return image, label
    
    def pad_image(self, image, label):
        # Ensure all image has the same resolution through padding
        w, h = image.size
        if w == h:
            padded = image
        elif w < h:
            # print("Smaller width")
            padding = (h - w) // 2
            padded = Image.new('RGB', (h, h))
            padded.paste(image, (padding, 0))
            label[:, 1] = (label[:, 1] * w + padding) / h
            label[:, 3] = label[:, 3] * w / h
        else:
            # print("Smaller height")
            padding = (w - h) // 2
            padded = Image.new('RGB', (w, w))
            padded.paste(image, (0, padding))
            label[:, 2] = (label[:, 2] * h + padding) / w
            label[:, 4] = label[:, 4] * h / w
        resize = transforms.Resize((self.imgSize, self.imgSize))
        image = resize(padded)
        return image, label

    def pad_lab(self, lab):
        pad_size = self.labSize - lab.shape[0]
        if(pad_size > 0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab

    def filter(self):
        print(f"All boxes should be larger than {self.imgSize * self.minBox}")
        if self.minBox == 0:
            return
        initialLength = len(self)
        for i in range(len(self)):
            index = initialLength - 1 - i
            image, label = self.__getitem__(index)
            for box in label:
                if box[3] < self.minBox or box[4] <self.minBox:
                    # print(f"{self.img_paths[index]} is removed")
                    self.img_paths.pop(index)
                    self.lab_paths.pop(index)
                    self.len -= 1
                    break
        print(f"After filtering, {len(self)} images remains")
                    
            


