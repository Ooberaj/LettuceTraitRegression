import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import cv2
from data import *
from model import *
from loss import *

def test_one_batch(inputs):
  # test in batches and functionalize due to limited memory
  with torch.no_grad():
    inputs = inputs.to(device)
    outputs = model(inputs)  
  return outputs.cpu()

def evaluate():
  means = [0.5482, 0.4620, 0.3602, 0.0127] 
  stds = [0.1639, 0.1761, 0.2659, 0.0035] 

  # images are cropped since I already cropped entire dataset during preprocessing step
  transforms = A.Compose(
        [A.Resize(320, 280, interpolation=cv2.INTER_LINEAR, p=1),
         A.Normalize(means, stds, max_pixel_value=1.0, p=1)
         ])
  testset = AutoGreenhouseChallenge(anno_file='drive/MyDrive/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/GroundTruth/GroundTruth_All_388_Images.json',
                                        depth_dir='drive/MyDrive/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/test_cropped_depth_images',
                                        rgb_dir='drive/MyDrive/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/test_cropped_images', transform=transforms)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=0, collate_fn = throwawayExceptions, shuffle=False)
  mse = nn.MSELoss()
  NMSE_Loss = NMSELoss()

  device = torch.device('cuda')
  model = MidFusionSubnet()
  model = model.to(device)
  state_dict = torch.load('/content/drive/MyDrive/AutoGreenhouseChallenge/models/checkpoint.pth')
  model.load_state_dict(state_dict)
  model.train(False)

  with torch.no_grad():
    outputs = torch.zeros(0, 5)
    labels = torch.zeros(0, 5)
    for inputs, labels_ in test_loader:
        output = test_one_batch(inputs)
        outputs = torch.cat((outputs, output), 0)
        labels = torch.cat((labels, labels_.cpu()), 0)
    print("NMSE Loss: ", NMSE_Loss(outputs, labels).item())
    print("\nMSE:")
    print(" Fresh Weight: ", mse(outputs[:,0], labels[:,0]).item())
    print(" Dry Weight: ", mse(outputs[:,1], labels[:,1]).item())
    print(" Height: ", mse(outputs[:,2], labels[:,2]).item())
    print(" Diameter: ", mse(outputs[:,3], labels[:,3]).item())
    print(" Leaf Area: ", mse(outputs[:,4], labels[:,4]).item())

    ac1 = torch.mean(abs(outputs[:,0] - labels[:,0])/labels[:,0], 0)
    ac2 = torch.mean(abs(outputs[:,1] - labels[:,1])/labels[:,1], 0)
    ac3 = torch.mean(abs(outputs[:,2] - labels[:,2])/labels[:,2], 0)
    ac4 = torch.mean(abs(outputs[:,3] - labels[:,3])/labels[:,3], 0)
    ac5 = torch.mean(abs(outputs[:,4] - labels[:,4])/labels[:,4], 0)

    print("\nMean percent error: ")
    print(" Fresh Weight: ", ac1.item())
    print(" Dry Weight: ", ac2.item())
    print(" Height: ", ac3.item())
    print(" Diameter: ", ac4.item())
    print(" Leaf Area: ", ac5.item())