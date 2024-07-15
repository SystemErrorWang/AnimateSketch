import os
import cv2
import numpy as np
import pandas as pd
import json
import random
import torch
import torchvision.transforms as transforms

from PIL import Image
from einops import rearrange
from decord import VideoReader
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor



def sobel_detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    #return dst[:, :, np.newaxis]
    return np.stack([dst]*3, axis=2)


class EdgeImageDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_path=None,
        min_margin=10,
        max_margin=20,
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.min_margin = min_margin
        self.max_margin = max_margin
        '''
        self.path_list = list()
        for root, dirs, files in os.walk(data_path, topdown=False):
            for name in files:
                if '.mp4' in name:
                    path = os.path.join(root, name)
                    self.path_list.append(path)
        '''
        self.path_list = list(pd.read_csv(data_path).path)
        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ), transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),])

        self.cond_transform = transforms.Compose(
            [transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ), transforms.ToTensor(),])

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def process(self, video_path):
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        #margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - self.max_margin)
        tgt_img_idx = random.randint(ref_img_idx + self.min_margin, 
                                    ref_img_idx + self.max_margin)
    
        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_edge = sobel_detect(tgt_img.asnumpy())
        tgt_edge_pil = Image.fromarray(tgt_edge)

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_edge_img = self.augmentation(tgt_edge_pil, self.cond_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            img=tgt_img,
            tgt_edge=tgt_edge_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )

        return sample

    def __getitem__(self, index):
        try:
            video_path = self.path_list[index]
            sample = self.process(video_path)
        except:
            rand_idx = np.random.randint(len(self.path_list))
            video_path = self.path_list[rand_idx]
            sample = self.process(video_path)
        return sample


    def __len__(self):
        return len(self.path_list)
