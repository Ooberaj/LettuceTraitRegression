import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import copy

class EarlyFusionGAP(nn.Module):
  def __init__(self):
    super(EarlyFusionGAP, self).__init__()
    self.resNet1 = torchvision.models.resnet18(pretrained = True)
    self.resNet2 = torchvision.models.resnet18(pretrained = True)
    self.resNet3 = torchvision.models.resnet18(pretrained = True)
    self.resNet4 = torchvision.models.resnet18(pretrained = True)
    self.resNet5 = torchvision.models.resnet18(pretrained = True)

    with torch.no_grad():
      weight = self.resNet1.conv1.weight.clone()
      self.resNet1.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet1.conv1.weight[:, :3] = weight[:, :3]
      self.resNet1.conv1.weight[:, 3] = weight[:, 0]

      self.resNet2.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet2.conv1.weight[:, :3] = weight[:, :3]
      self.resNet2.conv1.weight[:, 3] = weight[:, 0]

      self.resNet3.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet3.conv1.weight[:, :3] = weight[:, :3]
      self.resNet3.conv1.weight[:, 3] = weight[:, 0]

      self.resNet4.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet4.conv1.weight[:, :3] = weight[:, :3]
      self.resNet4.conv1.weight[:, 3] = weight[:, 0]

      self.resNet4.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet4.conv1.weight[:, :3] = weight[:, :3]
      self.resNet4.conv1.weight[:, 3] = weight[:, 0]

      self.resNet5.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet5.conv1.weight[:, :3] = weight[:, :3]
      self.resNet5.conv1.weight[:, 3] = weight[:, 0]

  def forward(self, x):
    out1 = torch.unsqueeze(torch.mean(self.resNet1(x), 1), 1)
    out2 = torch.unsqueeze(torch.mean(self.resNet2(x), 1), 1)
    out3 = torch.unsqueeze(torch.mean(self.resNet3(x), 1), 1)
    out4 = torch.unsqueeze(torch.mean(self.resNet4(x), 1), 1)
    out5 = torch.unsqueeze(torch.mean(self.resNet5(x), 1), 1)
    outs = torch.cat([out1, out2, out3, out4, out5], dim = 1);
    return outs

class EarlyFusionFC(nn.Module):
  def __init__(self):
    super(EarlyFusionFC, self).__init__()
    self.resNet1 = torchvision.models.resnet18(pretrained = True)
    self.resNet2 = torchvision.models.resnet18(pretrained = True)
    self.resNet3 = torchvision.models.resnet18(pretrained = True)
    self.resNet4 = torchvision.models.resnet18(pretrained = True)
    self.resNet5 = torchvision.models.resnet18(pretrained = True)

    with torch.no_grad():
      weight = self.resNet1.conv1.weight.clone()
      self.resNet1.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet1.conv1.weight[:, :3] = weight[:, :3]
      self.resNet1.conv1.weight[:, 3] = weight[:, 0]

      self.resNet2.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet2.conv1.weight[:, :3] = weight[:, :3]
      self.resNet2.conv1.weight[:, 3] = weight[:, 0]

      self.resNet3.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet3.conv1.weight[:, :3] = weight[:, :3]
      self.resNet3.conv1.weight[:, 3] = weight[:, 0]

      self.resNet4.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet4.conv1.weight[:, :3] = weight[:, :3]
      self.resNet4.conv1.weight[:, 3] = weight[:, 0]

      self.resNet4.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet4.conv1.weight[:, :3] = weight[:, :3]
      self.resNet4.conv1.weight[:, 3] = weight[:, 0]

      self.resNet5.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.resNet5.conv1.weight[:, :3] = weight[:, :3]
      self.resNet5.conv1.weight[:, 3] = weight[:, 0]
    
      self.net1 = nn.Sequential(self.resNet1, nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
      self.net2 = nn.Sequential(self.resNet2, nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
      self.net3 = nn.Sequential(self.resNet3, nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
      self.net4 = nn.Sequential(self.resNet4, nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
      self.net5 = nn.Sequential(self.resNet5, nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())

  def forward(self, x):
    out1 = self.net1(x)
    out2 = self.net2(x)
    out3 = self.net3(x)
    out4 = self.net4(x)
    out5 = self.net5(x)
    outs = torch.cat([out1, out2, out3, out4, out5], dim = 1)
    return outs

class MidFusion(nn.Module):
  def __init__(self):
    super(MidFusion, self).__init__()
    self.resNet18 = torchvision.models.resnet18(pretrained = True)
    self.depth_encoder = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.fcn = nn.Sequential(nn.Linear(2000, 1000), nn.ReLU(), nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 5), nn.ReLU())
  def forward(self, x):
    rgb_features = self.resNet18(x[:, :3, :, :])
    depth_features = self.depth_encoder(x[:, 3: , :, :])
    x = torch.cat([rgb_features, depth_features], dim = 1);
    x = self.fcn(x)
    return x

class MidFusionSubnet(nn.Module):
  def __init__(self):
    super(MidFusionSubnet, self).__init__()
    self.resNet1rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet2rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet3rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet4rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet5rgb = torchvision.models.resnet18(pretrained = True)
    self.depthEncoder1 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder2 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder3 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder4 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder5 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))

    dropout_p = 0.05
    self.fcn1 = nn.Sequential(nn.Dropout(dropout_p), nn.ReLU(), nn.Linear(2000, 1000), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(1000, 500), nn.Dropout(dropout_p), nn.ReLU(), nn.Linear(500, 1), nn.ReLU())
    self.fcn2 = nn.Sequential(nn.Dropout(dropout_p), nn.ReLU(), nn.Linear(2000, 1000), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(1000, 500), nn.Dropout(dropout_p), nn.ReLU(), nn.Linear(500, 1), nn.ReLU())
    self.fcn3 = nn.Sequential(nn.Dropout(dropout_p), nn.ReLU(), nn.Linear(2000, 1000), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(1000, 500), nn.Dropout(dropout_p), nn.ReLU(), nn.Linear(500, 1), nn.ReLU())
    self.fcn4 = nn.Sequential(nn.Dropout(dropout_p), nn.ReLU(), nn.Linear(2000, 1000), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(1000, 500), nn.Dropout(dropout_p), nn.ReLU(), nn.Linear(500, 1), nn.ReLU())
    self.fcn5 = nn.Sequential(nn.Dropout(dropout_p), nn.ReLU(), nn.Linear(2000, 1000), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(1000, 500), nn.Dropout(dropout_p), nn.ReLU(), nn.Linear(500, 1), nn.ReLU())

  def forward(self, x):
    rgbIn = x[:, :3, :, :]
    depthIn = x[:, 3: , :, :]
    out1 = self.fcn1(torch.cat([self.resNet1rgb(rgbIn), self.depthEncoder1(depthIn)], dim = 1))
    out2 = self.fcn2(torch.cat([self.resNet2rgb(rgbIn), self.depthEncoder2(depthIn)], dim = 1))
    out3 = self.fcn3(torch.cat([self.resNet3rgb(rgbIn), self.depthEncoder3(depthIn)], dim = 1))
    out4 = self.fcn4(torch.cat([self.resNet4rgb(rgbIn), self.depthEncoder4(depthIn)], dim = 1))
    out5 = self.fcn5(torch.cat([self.resNet5rgb(rgbIn), self.depthEncoder5(depthIn)], dim = 1))
    outs = torch.cat([out1, out2, out3, out4, out5], dim = 1)
    return outs

class MidFusionBatchNorm(nn.Module):
  def __init__(self):
    super(MidFusionBatchNorm, self).__init__()
    self.resNet1rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet2rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet3rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet4rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet5rgb = torchvision.models.resnet18(pretrained = True)
    self.depthEncoder1 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder2 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder3 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder4 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder5 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))

    self.fcn1 = nn.Sequential(nn.BatchNorm1d(2000), nn.ReLU(), nn.Linear(2000, 1000), nn.BatchNorm1d(1000), nn.ReLU(), nn.BatchNorm1d(1000), nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Linear(500, 1), nn.ReLU())
    self.fcn2 = nn.Sequential(nn.BatchNorm1d(2000), nn.ReLU(), nn.Linear(2000, 1000), nn.BatchNorm1d(1000), nn.ReLU(), nn.BatchNorm1d(1000), nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Linear(500, 1), nn.ReLU())
    self.fcn3 = nn.Sequential(nn.BatchNorm1d(2000), nn.ReLU(), nn.Linear(2000, 1000), nn.BatchNorm1d(1000), nn.ReLU(), nn.BatchNorm1d(1000), nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Linear(500, 1), nn.ReLU())
    self.fcn4 = nn.Sequential(nn.BatchNorm1d(2000), nn.ReLU(), nn.Linear(2000, 1000), nn.BatchNorm1d(1000), nn.ReLU(), nn.BatchNorm1d(1000), nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Linear(500, 1), nn.ReLU())
    self.fcn5 = nn.Sequential(nn.BatchNorm1d(2000), nn.ReLU(), nn.Linear(2000, 1000), nn.BatchNorm1d(1000), nn.ReLU(), nn.BatchNorm1d(1000), nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Linear(500, 1), nn.ReLU())

  def forward(self, x):
    rgbIn = x[:, :3, :, :]
    depthIn = x[:, 3: , :, :]
    out1 = self.fcn1(torch.cat([self.resNet1rgb(rgbIn), self.depthEncoder1(depthIn)], dim = 1))
    out2 = self.fcn2(torch.cat([self.resNet2rgb(rgbIn), self.depthEncoder2(depthIn)], dim = 1))
    out3 = self.fcn3(torch.cat([self.resNet3rgb(rgbIn), self.depthEncoder3(depthIn)], dim = 1))
    out4 = self.fcn4(torch.cat([self.resNet4rgb(rgbIn), self.depthEncoder4(depthIn)], dim = 1))
    out5 = self.fcn5(torch.cat([self.resNet5rgb(rgbIn), self.depthEncoder5(depthIn)], dim = 1))
    outs = torch.cat([out1, out2, out3, out4, out5], dim = 1)
    return outs

class LateFusionSum(nn.Module):
  def __init__(self):
    super(LateFusionSum, self).__init__()
    self.resNet18 = torchvision.models.resnet18(pretrained = True)
    self.depth_encoder = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.fcn_rgb = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 5), nn.ReLU())
    self.fcn_depth = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 5), nn.ReLU())

  def forward(self, x):
    rgb_features = self.resNet18(x[:, :3, :, :])
    depth_features = self.depth_encoder(x[:, 3: , :, :])
    rgb_out = self.fcn_rgb(rgb_features)
    depth_out = self.fcn_depth(depth_features)
    out = rgb_out + depth_out
    return out

class LateFusionAvg(nn.Module):
  def __init__(self):
    super(LateFusionAvg, self).__init__()
    self.resNet18 = torchvision.models.resnet18(pretrained = True)
    self.depth_encoder = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.fcn_rgb = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 5), nn.ReLU())
    self.fcn_depth = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 5), nn.ReLU())

  def forward(self, x):
    rgb_features = self.resNet18(x[:, :3, :, :])
    depth_features = self.depth_encoder(x[:, 3: , :, :])
    rgb_out = self.fcn_rgb(rgb_features)
    depth_out = self.fcn_depth(depth_features)
    out = (rgb_out + depth_out)/2
    return out

class LateFusionFcn(nn.Module):
  def __init__(self):
    super(LateFusionFcn, self).__init__()
    self.resNet18 = torchvision.models.resnet18(pretrained = True)
    self.depth_encoder = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.fcn_rgb = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 5), nn.ReLU())
    self.fcn_depth = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 5), nn.ReLU())
    self.fcn_fusion = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 50), nn.Linear(50, 5), nn.ReLU())

  def forward(self, x):
    rgb_features = self.resNet18(x[:, :3, :, :])
    depth_features = self.depth_encoder(x[:, 3: , :, :])
    rgb_out = self.fcn_rgb(rgb_features)
    depth_out = self.fcn_depth(depth_features)
    out = self.fcn_fusion(torch.cat([rgb_out, depth_out], 1))
    return out

class LateFusionSubnet(nn.Module):
  def __init__(self):
    super(LateFusionSubnet, self).__init__()
    self.resNet1rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet2rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet3rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet4rgb = torchvision.models.resnet18(pretrained = True)
    self.resNet5rgb = torchvision.models.resnet18(pretrained = True)
    self.depthEncoder1 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder2 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder3 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder4 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))
    self.depthEncoder5 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), torchvision.models.resnet18(pretrained = True))

    self.fcn1_rgb = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
    self.fcn2_rgb = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
    self.fcn3_rgb = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
    self.fcn4_rgb = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
    self.fcn5_rgb = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
    self.fcn1_depth = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
    self.fcn2_depth = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
    self.fcn3_depth = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
    self.fcn4_depth = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())
    self.fcn5_depth = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(), nn.Linear(250, 1), nn.ReLU())

  def forward(self, x):
    rgbIn = x[:, :3, :, :]
    depthIn = x[:, 3: , :, :]

    out1_rgb = self.fcn1_rgb(self.resNet1rgb(rgbIn))
    out2_rgb = self.fcn2_rgb(self.resNet2rgb(rgbIn))
    out3_rgb = self.fcn3_rgb(self.resNet3rgb(rgbIn))
    out4_rgb = self.fcn4_rgb(self.resNet4rgb(rgbIn))
    out5_rgb = self.fcn5_rgb(self.resNet5rgb(rgbIn))
    
    out1_depth = self.fcn1_depth(self.depthEncoder1(depthIn))
    out2_depth = self.fcn2_depth(self.depthEncoder2(depthIn))
    out3_depth = self.fcn3_depth(self.depthEncoder3(depthIn))
    out4_depth = self.fcn4_depth(self.depthEncoder4(depthIn))
    out5_depth = self.fcn5_depth(self.depthEncoder5(depthIn))

    out1 = out1_rgb + out1_depth
    out2 = out2_rgb + out2_depth
    out3 = out3_rgb + out3_depth
    out4 = out4_rgb + out4_depth
    out5 = out5_rgb + out5_depth

    outs = torch.cat([out1, out2, out3, out4, out5], dim = 1)
    return outs

class comboWombo(nn.Module):
  def __init__(self):
    super(comboWombo, self).__init__()
    self.MidFusion = MidFusion()
    self.MidFusionSubnet = MidFusionSubnet()

  def forward(self, x):
    out_single = self.MidFusion(x)
    out_subnet = self.MidFusionSubnet(x)
    out = (out_single + out_subnet)/2
    return out