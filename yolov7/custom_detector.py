import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from yolov7.models.experimental import attempt_load
from models.yolo import Model
from utils.general import non_max_suppression

class Detector():
    def __init__(self, weights):
        self.model = attempt_load(weights, map_location="cuda")
        # print(self.model)
        # with open("yolov7/data/hyp.scratch.p5.yaml") as f:
        #     hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        # ckpt = torch.load(weights, map_location="cuda")
        # self.model = Model(ckpt['model'].yaml, ch=3, nc=80, anchors=hyp.get('anchors')).to("cuda")
        # state_dict = ckpt['model'].float().state_dict() 
        # self.model.load_state_dict(state_dict, strict=False)  # load
    
    def detect(self, images, conf_thres=0.5, classes=None):
        pred = self.model(images, augment=False)
        pred = pred[0]
        pred = non_max_suppression(pred, conf_thres, 0.4, classes=classes)
        # print(type(pred))
        # print(pred[0])
        return pred

        