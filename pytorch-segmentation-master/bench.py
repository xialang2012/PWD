import os
import json
import argparse
import torch
import models
import inspect
import math
from utils import losses
from utils import Logger
from utils.torchsummary import summary


from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler

try:
    from osgeo import gdal, osr, ogr
except ImportError:
    import gdal, osr, ogr

import time

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import torch.nn.functional as F
import time

from torchmetrics.functional import f1
from torchmetrics import IoU

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class ContrailDataset(Dataset):
    def __init__(self, r_dir, gt_dir, pytorch=True):
        super().__init__()

        self.t = np.zeros((256, 256))
        self.m = np.zeros((256, 256))

        self.files = [self.combine_files(f, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
    
    def combine_files(self, r_file:Path, gt_dir):
        files = {   'band': r_file,
                    'gt': gt_dir/r_file.name    }

        return files
    
    def __len__(self):
        return len(self.files)

    def LoadRaster(self, rasterFile, bandNum=1):           

        # Open the file:
        dataset = gdal.Open(rasterFile)
        band = dataset.GetRasterBand(bandNum).ReadAsArray()
        geoTrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        rows, cols = band.shape

        if dataset.RasterCount == 1:
            return band

        band = band.reshape((rows, cols, 1))
        for i in range(2, dataset.RasterCount+1):
            tmpBand = dataset.GetRasterBand(i).ReadAsArray().reshape((rows, cols, 1))
            band = np.concatenate([band, tmpBand], axis = 2)              

        return band/255

    def open_as_array(self, idx, invert=False, include_nir=False):

        raw = np.transpose(self.LoadRaster(str(self.files[idx]['band'])), (2,0,1))
        #raw = np.expand_dims(raw, axis=0)
        #raw = np.expand_dims(self.t, axis=0)
        
        return raw

    def open_mask(self, idx, add_dims=True):

        #raw_mask = np.array(Image.open(self.files[idx]['gt']))
        #k = str(self.files[idx]['gt'])
        raw_mask = self.LoadRaster(str(self.files[idx]['gt']))
        raw_mask[raw_mask > 0] = 1
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, idx):
        
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.float32)
        
        return x, y
    
    def open_as_pil(self, idx):
        
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s


def dice(input, target):
    input = torch.round(torch.sigmoid(input))
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def iou(input, target):
    input = torch.round(torch.sigmoid(input))
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    epsilon = 1e-7
    return ((intersection) / (iflat.sum() + tflat.sum() - intersection + epsilon))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()

def f1_metrics(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=False) -> torch.Tensor:

    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    #f1.requires_grad = is_training
    return f1, precision, recall

def acc_metric(y_pred, y_test):    

    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.numel()
    return acc.cpu()


def SaveRaster(fileName, proj, geoTrans, data):

    # type
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # check shape of array
    if len(data.shape) == 3:
        im_height, im_width, im_bands = data.shape
    else:
        im_bands, (im_height, im_width) = 1, data.shape 

    # create file
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fileName, im_width, im_height, im_bands, datatype, options=['COMPRESS=LZW'])
    if len(geoTrans) == 6:
        dataset.SetGeoTransform(geoTrans)
    if len(proj) > 0:
        dataset.SetProjection(proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(data[:, :, i])

def LoadRaster(rasterFile, bandNum=1):           

    # Open the file:
    dataset = gdal.Open(rasterFile)
    band = dataset.GetRasterBand(bandNum).ReadAsArray()
    geoTrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    rows, cols = band.shape

    if dataset.RasterCount == 1:
        return band

    band = band.reshape((rows, cols, 1))
    for i in range(2, dataset.RasterCount+1):
        tmpBand = dataset.GetRasterBand(i).ReadAsArray().reshape((rows, cols, 1))
        band = np.concatenate([band, tmpBand], axis = 2)              

    return band / 255

       
class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

def fit(model, train_dl, loss_fn, optimizer, epoch, args):
    
    best_f1 = 0
    model.train()

    iou = IoU(num_classes=args.n_class).to(args.device)

    for batch_idx, (data, target) in enumerate(train_dl):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)[0]
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

        if args.n_class == 1:
            y_pred_tag = torch.round(torch.sigmoid(outputs))
        else:
            y_pred_tag = F.softmax(outputs, dim=1).argmax(dim=1)

        f1_torch = f1(y_pred_tag, target, average ='micro', num_classes=args.n_class, mdmc_average ='global')
        iou_torch = iou(y_pred_tag, target)
        acc = 0
                            
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} Acc {:.4f} iou {:.4f} f1 {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_dl.dataset),
                100. * batch_idx / len(train_dl), loss.item(), acc, iou_torch, f1_torch))

def test(model, device, test_loader, loss_fn, args):

    model.eval()
    test_loss, test_acc, test_f1, test_iou = 0, 0, 0, 0
    correct = 0
    iou = IoU(num_classes=args.n_class).to(args.device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[0]
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            
            y_pred_tag = F.softmax(output, dim=1).argmax(dim=1)

            test_acc += 0
            test_f1 += f1(y_pred_tag, target, average ='micro', num_classes=args.n_class, mdmc_average ='global')
            test_iou += iou(y_pred_tag, target)

    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset) * args.batch_size
    test_f1 /= len(test_loader.dataset) * args.batch_size
    test_iou /= len(test_loader.dataset) * args.batch_size

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) Acc {:.4f} iou {:.4f} f1 {:.4f}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), test_acc, test_iou, test_f1))

    accs = {'acc': test_acc,
            'f1': test_f1,
             'iou': test_iou}
    return accs

def train(model, train_dl, valid_dl, loss_fn, opt, args):    
    
    start = time.time()
    best_f1 = 0

    for epoch in range(1, args.epochs + 1):

        fit(model, train_dl, loss_fn, opt, epoch, args)

        accs = test(model, args.device, valid_dl, loss_fn, args)


        if accs.f1 > best_f1:

            best_f1 = accs.f1

            torch.save(model.state_dict(), args.modelName + '_epoch_' + str(epoch) + '.pth')
            with open(modelName + '_perf.txt', 'a', encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ', ' + str(epoch) + ', ' + str(args.acc) + ', '+ str(args.iou) + ', ' + str(args.f1) + '\n')

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 

torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = Path(r'../train')
data = ContrailDataset(base_path/'image', base_path/'mask')
print(len(data))


train_ds, valid_ds = torch.utils.data.random_split(data, (int(len(data)*0.75), len(data) - int(len(data)*0.75)))
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, pin_memory=True, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=2, shuffle=True, pin_memory=True, drop_last=True)


config = json.load(open(r'./pytorch-segmentation-master/config.json'))
modelName = 'SegNet'  # change to 'Unet' to run the unet
config['arch']['type']  = modelName
config['arch']['args']['backbone'] = 'resnet50'
model = get_instance(models, 'arch', config, 1)
print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))

epochs = 100
batch_size = 8
train_kwargs = {'batch_size': batch_size, 
                'epochs': epochs,
                'model_name': modelName,
                'device': device,
                'n_class': 3,
                'log_interval': 100}

train_kwargs = DotDict(train_kwargs)
model.to(train_kwargs.device)

#loss_fn = FocalLoss(2)
loss_fn = DiceLoss()

opt = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, train_dl, valid_dl, loss_fn, opt, train_kwargs)