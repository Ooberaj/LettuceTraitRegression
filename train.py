import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
import cv2
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure
import numpy as np
import os,sys,humanize,psutil,GPUtil
from data import *
from loss import *
from model import *

def plot_grad_flow(named_parameterss):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    Source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for named_parameters in named_parameterss:
      for n, p in named_parameters:
          if(p.requires_grad) and ("bias" not in n):
              if(p.grad is not None):
                layers.append(n)
                ave_grads.append(p.grad.cpu().abs().mean())
                max_grads.append(p.grad.cpu().abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def mem_report():
  print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))


def load_data():
  means = [0.5482, 0.4620, 0.3602, 0.0127] 
  stds = [0.1639, 0.1761, 0.2659, 0.0035] 

  transforms = A.Compose(
      [A.Resize(320, 280, interpolation=cv2.INTER_LINEAR, p=1),
       A.Normalize(means, stds, max_pixel_value=1.0, p=1)
       ])

  dataset = AutoGreenhouseChallenge(anno_file='drive/MyDrive/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/GroundTruth/GroundTruth_All_388_Images.json',
                                        depth_dir='drive/MyDrive/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/train_cropped_depth_images',
                                        rgb_dir='drive/MyDrive/AutoGreenhouseChallenge/autonomous_greenhouse_challenge/train_cropped_images', transform=transforms)
  train, val, test = train_val_test_split(dataset, train_perc = 0.8, val_perc = 0.2, test_perc = 0)
  train_loader = torch.utils.data.DataLoader(train, batch_size= 15, num_workers=0, shuffle=True, collate_fn = throwawayExceptions, drop_last=True)
  print(len(train_loader))
  val_loader = torch.utils.data.DataLoader(val, batch_size=5, num_workers=0, shuffle=True, collate_fn = throwawayExceptions, drop_last=True)

  return train_loader, val_loader

def train_one_batch(inputs, labels):
  # functionalize to create new variables for each new example in a batch and
  # free unused variables of previous batches to reduce memory usage
  inputs, labels = inputs.to(device), labels.to(device)
  mem_report()
  
  outputs = model(inputs)
  mem_report()
  
  loss = NMSE_Loss(outputs, labels)
  
  optimizer.zero_grad()
  loss.backward()
  plot_grad_flow([model.fcn1.named_parameters(), model.fcn2.named_parameters(), model.fcn3.named_parameters(), model.fcn4.named_parameters(), model.fcn5.named_parameters()])
  optimizer.step()
  
  mem_report()

  return loss

def train_one_epoch(epoch, tb_writer, loader):
  mem_report() 
  running_loss = 0
  for i, data in enumerate(loader):
    inputs, labels = data
    batch_loss = train_one_batch(inputs, labels)
    running_loss += batch_loss
    # For each batch, display its loss
    print('  batch {} loss: {}'.format(i + 1, batch_loss))
    tb_x = epoch * len(loader) + i + 1
    # Write loss per batch = total loss / # batches processed so far
    tb_writer.add_scalar('Loss vs. batch #', batch_loss, tb_x)
  return running_loss/len(loader)

def validate_one_batch(inputs, labels):
  inputs, labels = inputs.to(device), labels.to(device)
  outputs = model(inputs)  
  loss = NMSE_Loss(outputs, labels)
  return loss.item()

def validate(epoch, tb_writer, loader):
  running_loss = 0.0
  for i, val_data in enumerate(loader):
    val_inputs, val_labels = val_data
    running_loss += validate_one_batch(val_inputs, val_labels)
  return running_loss/len(loader)

def train():
  train_loader, val_loader = load_data()

  # Write output to ./runs/lettuce_trainer/timestamp directory
  timestamp = datetime.now().strftime('%H%M%S')
  writer = SummaryWriter('logs/lettuce_trainer_{}'.format(timestamp))

  # Initialize model
  device = torch.device('cuda')
  mem_report() 
  model = MidFusionSubnet()
  model = model.to(device)
  NMSE_Loss = NMSELoss()
  mem_report() 

  optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, maximize=False)   
  EPOCHS = 100

  best_val_loss = 99999999
  for epoch in range(EPOCHS):
      print('EPOCH {}:'.format(epoch + 1))

      model.train(True)
      avg_train_loss = train_one_epoch(epoch, writer, train_loader)
      
      print("Validation")    
      model.train(False)
      avg_val_loss = validate(epoch, writer, val_loader)

      writer.add_scalars('Avg Train Loss vs Validation Loss',
                      { 'Training' : avg_train_loss, 'Validation' : avg_val_loss },
                      epoch + 1)
      writer.flush()

      print('LOSS Validation {}'.format(avg_val_loss))

      # Save state of best model
      if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          model_save_name = "model_" + str(timestamp) + "_" + str(epoch)
          model_path = F"/content/drive/MyDrive/AutoGreenhouseChallenge/models/checkpointDropIncrease.pth"
          torch.save(model.state_dict(), model_path)
          print("saved")