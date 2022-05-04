import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import (Dataset, dataloader)
from torchvision.io import read_image
import albumentations as A
import cv2
from PIL import Image
import numpy as np
import random
import json

class AutoGreenhouseChallenge(Dataset):
    def __init__(self, anno_file, depth_dir, rgb_dir, transform=None):
        annotations = pd.read_json(anno_file)
        self.img_labels = annotations.iloc[4:, 1]
        self.depth_dir = depth_dir
        self.rgb_dir = rgb_dir
        self.rgb_names = os.listdir(rgb_dir)
        self.depth_names = os.listdir(depth_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rgb_names)

    def __getitem__(self, idx):
        img_name = self.rgb_names[idx]

        try:
            if img_name[:3] == "Aug":
                anno_name = "AugImage" + img_name.split('_')[1].split('.')[0]
            else:
                anno_name = "Image" + img_name.split('_')[1].split('.')[0]
            img_data = self.img_labels.loc[anno_name]
        except Exception as e:
            print(e)
            return None

        rgb_img_path = os.path.join(self.rgb_dir, img_data['RGB_Image'])
        depth_img_path = os.path.join(self.depth_dir, img_data['Depth_Information'])

        try:
            rgb_img = plt.imread(rgb_img_path)
            depth_img = plt.imread(depth_img_path)
            rgbd_img = np.dstack([rgb_img, depth_img])
        except Exception as e:
            print(e)
            return None

        labels = [img_data['FreshWeightShoot'], img_data['DryWeightShoot'], img_data['Height'], img_data['Diameter'], img_data['LeafArea']]
        labels = torch.as_tensor(labels, dtype=torch.float32)

        if self.transform:
            aug = self.transform(image = rgbd_img)
            rgbd_img = aug['image']

        rgbd_img = np.transpose(rgbd_img, (2, 1, 0))
        rgbd_img = torch.as_tensor(rgbd_img, dtype=torch.float32)

        return rgbd_img, labels

def throwawayExceptions(batch):
    batch = filter(lambda data: data is not None, batch)
    return dataloader.default_collate(list(batch))

def getMeanStd(image_loader):
    sum_means, squared_sum, num_batches = 0, 0, 0
    for image in tqdm(image_loader):
        # operate along batch, height, and width axes
        sum_means += torch.mean(image, dim=[0, 2, 3])
        squared_sum += torch.mean(image**2, dim=[0, 2, 3])
        num_batches += 1

    mean = sum_means / num_batches
    std = (squared_sum / num_batches - mean ** 2) ** 0.5

    torch.set_printoptions(precision=10)
    mean, std = getMeanStd(loader)
    print(mean, std)

    return mean, std

def crop():
    min_x=650
    max_x=1450
    min_y=200
    max_y=900
    Depth_Data_Dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/DepthImages'
    RGB_Data_Dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/RGBImages'
    cropped_img_dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/cropped_images/'
    cropped_depth_img_dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/cropped_depth_images/'

    os.mkdir(cropped_img_dir)
    os.mkdir(cropped_depth_img_dir)

    for im in os.listdir(RGB_Data_Dir):
        img = cv2.imread(os.path.join(RGB_Data_Dir,im))
        crop_img = img[min_y:max_y,min_x:max_x]
        cv2.imwrite(os.path.join(cropped_img_dir,im), crop_img)

    for depth_im in os.listdir(Depth_Data_Dir):
        depth_img = cv2.imread(os.path.join(Depth_Data_Dir,depth_im), 0)
        crop_depth_img = depth_img[min_y:max_y,min_x:max_x]
        cv2.imwrite(os.path.join(cropped_depth_img_dir,depth_im), crop_depth_img)

    RGB_Data_Dir = cropped_img_dir
    Depth_Data_Dir = cropped_depth_img_dir

def train_val_test_split(dataset, train_perc, val_perc, test_perc):
    train_size = int(len(dataset) * train_perc)
    val_size = int(len(dataset) * val_perc)
    test_size = len(dataset) - train_size - val_size
    train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    return train, val, test


def splitData():
    '''
        Crop images and put 50 rgb and depth images into a separate test directory.
    '''
    min_x=650
    max_x=1450
    min_y=200
    max_y=900

    nums = set()
    Rgb_names = []
    depth_names = []
    while(len(nums) != 51):
        nums.add(int(random.uniform(1, 392)))
    for num in nums:
        Rgb_name = "RGB_" + str(num) + ".png"
        Rgb_names.append(Rgb_name)
        depth_name = "Depth_" + str(num) + ".png"
        depth_names.append(depth_name)

    os.mkdir(train_cropped_img_dir)
    os.mkdir(train_cropped_depth_img_dir)
    os.mkdir(test_cropped_img_dir)
    os.mkdir(test_cropped_depth_img_dir)

    for im in os.listdir(RGB_Data_Dir):
        if im in Rgb_names:
            try:
                img = cv2.imread(os.path.join(RGB_Data_Dir,im))
                crop_img = img[min_y:max_y,min_x:max_x]
                cv2.imwrite(os.path.join(test_cropped_img_dir,im), crop_img)
            except Exception as e:
                print(e)
                continue
        else:
            try:
                img = cv2.imread(os.path.join(RGB_Data_Dir,im))
                crop_img = img[min_y:max_y,min_x:max_x]
                cv2.imwrite(os.path.join(train_cropped_img_dir,im), crop_img)
            except Exception as e:
                print(e)
                continue

    for depth_im in os.listdir(Depth_Data_Dir):
        if depth_im in depth_names:
            try:
                img = cv2.imread(os.path.join(Depth_Data_Dir,depth_im), 0)
                crop_img = img[min_y:max_y,min_x:max_x]
                cv2.imwrite(os.path.join(test_cropped_depth_img_dir,depth_im), crop_img)
            except Exception as e:
                print(e)
                continue
        else:
            try:
                img = cv2.imread(os.path.join(Depth_Data_Dir,depth_im), 0)
                crop_img = img[min_y:max_y,min_x:max_x]
                cv2.imwrite(os.path.join(train_cropped_depth_img_dir,depth_im), crop_img)
            except Exception as e:
                print(e)
                continue


def augment():
    RGB_Data_Dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/train_cropped_images/'
    Depth_Data_Dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/train_cropped_depth_images/'
    RGB_Data_Aug_Dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/train_aug_cropped_images/'
    Depth_Data_Aug_Dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/train_aug_cropped_depth_images/'

    os.mkdir(RGB_Data_Aug_Dir)
    os.mkdir(Depth_Data_Aug_Dir)
    means = [0.5482, 0.4620, 0.3602, 0.0127] 
    stds = [0.1639, 0.1761, 0.2659, 0.0035] 

    for im, depth_im in zip(os.listdir(RGB_Data_Dir), os.listdir(Depth_Data_Dir)):
        transforms = A.Compose(
            [A.Flip(p = 0.5),
             A.ShiftScaleRotate(shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-45, 45), 
                                interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=means, 
                                mask_value= None, p=1)
             ])
        try: 
            img = cv2.imread(os.path.join(RGB_Data_Dir,im))
            if img is None:
                continue
        except Exception as e:
            print(e)
            continue
        transformed = transforms(image = img) 
        transformed_img = transformed["image"] 
        im = "Aug" + im
        cv2.imwrite(os.path.join(RGB_Data_Aug_Dir,im), transformed_img)

        depth_img = cv2.imread(os.path.join(Depth_Data_Dir,depth_im), 0)
        transformed = transforms(image = depth_img) 
        transformed_img = transformed["image"] 
        depth_im = "Aug" + depth_im 
        cv2.imwrite(os.path.join(Depth_Data_Aug_Dir,depth_im), transformed_img)

def addAugAnnotations():
    anno_file='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/GroundTruth/GroundTruth_All_388_Images.json'
    annotations = pd.read_json(anno_file)
    img_labels = annotations.iloc[4:]

    with open(anno_file, 'r+') as f:
        data = json.load(f)
        for name, row in img_labels.iterrows():
            img_label = row['Measurements']
            labels = {
                "Variety" : img_label['Variety'], 
                "RGB_Image" : "Aug" + img_label['RGB_Image'], 
                "Depth_Information" : "Aug" + img_label['Depth_Information'],
                "FreshWeightShoot": img_label['FreshWeightShoot'], 
                "DryWeightShoot": img_label['DryWeightShoot'], 
                "Height": img_label['Height'], 
                "Diameter": img_label['Diameter'], 
                "LeafArea": img_label['LeafArea']
                }
            data["Measurements"]["Aug" + name] = labels
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

Depth_Data_Dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/DepthImages'
RGB_Data_Dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/RGBImages'
train_cropped_img_dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/train_cropped_images/'
train_cropped_depth_img_dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/train_cropped_depth_images/'
test_cropped_img_dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/test_cropped_images/'
test_cropped_depth_img_dir='/Users/yuvrajvirk/Desktop/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/test_cropped_depth_images/'