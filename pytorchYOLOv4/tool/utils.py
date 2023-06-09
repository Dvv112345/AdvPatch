import sys
import os
import time
import math
import numpy as np
import torch

import itertools
import struct  # get_image_size
import imghdr  # get_image_size


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    
    # print('iou box1:', box1)
    # print('iou box2:', box2)

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort(descending=True)
    keep = []
    while order.shape[0] > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self.reshape(1))

        xx1 = torch.maximum(x1[idx_self], x1[idx_other])
        yy1 = torch.maximum(y1[idx_self], y1[idx_other])
        xx2 = torch.minimum(x2[idx_self], x2[idx_other])
        yy2 = torch.minimum(y2[idx_self], y2[idx_other])

        w = torch.maximum(torch.tensor([0.0]).cuda(), xx2 - xx1)
        h = torch.maximum(torch.tensor([0.0]).cuda(), yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / torch.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = torch.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    if len(keep) == 0:
        return torch.zeros([0])
    return torch.cat(keep)



def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, f"{class_names[cls_id]}: {cls_conf:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names



def post_processing(img, conf_thresh, nms_thresh, output, show_detail=False, targetClass=None):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # box_array, [batch, num, 1, 4]       , torch.Size([1, 22743, 1, 4])
    box_array = output[0]
    # confs,     [batch, num, num_classes], torch.Size([1, 22743, 80])
    confs = output[1]
    # print("Conf")
    # print(confs)
    max_det = 300

    t1 = time.time()

    # if type(box_array).__name__ != 'ndarray':
    #     box_array = box_array.cpu().detach().numpy()
    #     confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = torch.max(confs, dim=2)[0]
    max_id = torch.argmax(confs, dim=2)
    # print(confs.shape)
    # print(max_conf.shape)
    # print(max_id.shape)

    t2 = time.time()

    bboxes_batch = []

    for i in range(box_array.shape[0]):
       
        argwhere = max_conf[i] > conf_thresh
        # print(argwhere.shape)
        l_box_array = box_array[i][argwhere]
        l_max_conf = max_conf[i][argwhere]
        l_max_id = max_id[i][argwhere]

        # print(l_box_array.shape)
        # print(l_max_conf.shape)
        # print(l_max_id.shape)

        if l_box_array.shape[0] > max_det:
            l_box_array = l_box_array[:300, :]
            l_max_conf = l_max_conf[:300]
            l_max_id = l_max_id[:300]

        bboxes = []
        # nms for each class
        for j in range(num_classes):
            if targetClass is not None and j != targetClass:
                continue
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            # print("ll_max_conf")
            # print(ll_max_conf)

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
            if (keep.shape[0] > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep].unsqueeze(1)
                ll_max_id = ll_max_id[keep].unsqueeze(1)
                # print(ll_box_array.shape)
                # print(ll_max_conf.shape)
                # print(ll_max_id.shape)

                bboxes.append(torch.cat([ll_box_array, ll_max_conf, ll_max_id], dim=1))
                # print(bboxes.shape)
                # for k in range(ll_box_array.shape[0]):
                #     bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_id[k]])
        if len(bboxes) == 0:
            bboxes.append(torch.tensor([[0,0,0,0,0,0]], dtype=torch.float32).cuda())

        bboxes = torch.cat(bboxes, dim = 0)
        bboxes_batch.append(bboxes)

    # print(len(bboxes_batch))
    # print(bboxes_batch[0].shape)
    # output = torch.cat(bboxes_batch, dim=0)
    # print(output.shape)
    t3 = time.time()

    if(show_detail):
        print('-----------------------------------')
        print('       max and argmax : %f' % (t2 - t1))
        print('                  nms : %f' % (t3 - t2))
        print('Post processing total : %f' % (t3 - t1))
        print('-----------------------------------')
    
    # print("bboxes_batch")
    # print(bboxes_batch[0])
    return bboxes_batch
