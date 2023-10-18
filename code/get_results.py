import traceback
import pydicom
import os
from unet.unet_model import UNet
import torch.nn as nn
from scipy.stats import linregress
from vif import vif
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as rmse
from skimage.metrics import normalized_mutual_information as nmi
from functools import partial
import piq
import pandas as pd
import numpy as np
import pyiqa
from multiprocessing import Pool
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.unet = UNet(1,10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        return self.unet(x)
def compute_suv(file_name,base_reference_time=0):

    estimated = False
    f=pydicom.dcmread(file_name)

    try:
        weight_grams = float(f.PatientWeight)*1000
    except:
        traceback.print_exc()
        weight_grams = 75000
        estimated = True

    try:
        injected_dose = float(f.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        injected_dose_decay = injected_dose
    except:
        traceback.print_exc()
        decay = np.exp(-np.log(2)*(1.75*3600)/6588)
        injected_dose_decay = 420000000 * decay
        estimated = True
    return (f.pixel_array.astype(np.float32)*f.RescaleSlope+f.RescaleIntercept)*weight_grams/injected_dose_decay, estimated,f.FrameReferenceTime
from torch.utils.data import Dataset,DataLoader
from collections import deque
class EMA:
    def __init__(self,N=100):
        self.EMA = None
        self.N = N
        self.history = deque([],maxlen=N+1)
    def update(self,x):
        if self.EMA is None:
            self.EMA = x
        else:
            self.EMA = (2*x+(self.N-1)*self.EMA)/(self.N+1)
        self.history.append(self.EMA)
        if len(self.history)>self.N:
            self.history.popleft()
    def val(self):
        return self.EMA
    def isup(self):
        if self.history[-1]-self.history[0]>0:
            return True
        else:
            return False
class PETDataset(Dataset):
    def __init__(self,time_frame=(0,10,20,30,40,50,60,90,120,150,180,240,300,360,420,540,660,780,900,1200,1500,1800,2100,2400,2700,3000,3300,3600),numOfpatients=103):
        self.time_frame_len = len(time_frame)
        self.numOfPatients = numOfpatients
        self.slice_len = 71
        self.mapping = {time_frame[i]:i for i in range(len(time_frame))}
        self.t = np.arange(3901)
    def blood_func(self, a0, a1,a2,a3,a4,a5,b0,b1,b2):
        t = self.t/60
        cp = ( a0*t - a1 - a2 ) * np.exp(-a3*t) + \
                     a1 * np.exp(-a4*t) + \
                     a2 * np.exp(-a5*t)
        cp = cp*np.exp(-float(np.log(2)/109.8)*t)
        wholeblood = cp/(b0*np.exp(-b1*t)+b2)
        return wholeblood
    def __getitem__(self, item):
        patientID = item//self.slice_len
        SUV = np.load(f"data/{item}_SUV.npy")
        blood = np.load(f"data/{patientID}_blood.npy")
        Ki = np.load(f"data_ki/{item}_Ki.npy")
        intercept = np.load(f"data_ki/{item}_intercept.npy")
        return SUV,blood,Ki,intercept
    def __len__(self):
        return self.slice_len*self.numOfPatients
from PIL import Image
from torchvision.utils import make_grid
import torch
from matplotlib import cm
@torch.no_grad()
def save_image(tensor,fp,format = None,**kwargs):
    tensor = tensor.detach().cpu().clone()
    if kwargs.get('scale_each',False):
        for t in tensor:
            low = t.min()
            high = t.max()
            t.sub_(low).div_(max(high - low, 1e-5))
    else:
        if kwargs.get('normalize',False):
            low = tensor.min()
            high = tensor.max()
            tensor.sub_(low).div_(max(high - low, 1e-5))

    if kwargs.get('normalize',False):
        kwargs['normalize'] = False
        cmap_vol = np.apply_along_axis(cm.get_cmap(kwargs.get('cmap','gray')), 0, tensor.numpy()) # converts prediction to cmap!
        cmap_vol = torch.from_numpy(np.squeeze(cmap_vol))
        grid = make_grid(cmap_vol[:,:3,::], **kwargs,nrow=int(np.sqrt(cmap_vol.shape[0])))
    else:
        grid = make_grid(tensor, **kwargs,nrow=int(np.sqrt(tensor.shape[0])))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import datetime
from _history import History
import time
from itertools import chain
import torch.autograd as autograd
def MyCumtrapz(y,x,dim=0,initial=None):
    out = torch.cumulative_trapezoid(y,x=x,dim=dim)
    if initial is not None:
        tmp = torch.ones_like(torch.index_select(out, dim, torch.tensor(0,device=out.device)))*initial
        out = torch.cat([tmp,out],dim=dim)
    return out
def MySimpleconvolve(tensor,filter,scant,dim=0):
    dimsize = tensor.size(dim)
    shape = tensor.shape
    shape = list(shape)
    shape[dim]=len(scant)
    out = torch.zeros(tuple(shape)).to(tensor)
    tmpshape = shape.copy()
    shape[dim] = -1
    shape[:dim] = [1 for _ in range(dim)]
    shape[dim+1:] = [1 for _ in range(len(shape)-dim-1)]
    filter = torch.flip(filter.squeeze(),[0])
    filter = filter.view(*shape)
    shape[:dim] = [slice(0,None) for _ in range(dim)]
    shape[dim + 1:] = [slice(0,None) for _ in range(len(shape)-dim-1)]
    tmpshape[:dim] = [slice(0,None) for _ in range(dim)]
    tmpshape[dim + 1:] = [slice(0,None) for _ in range(len(shape)-dim-1)]
    outshape = tmpshape.copy()
    for i in range(len(scant)):
        shape[dim] = slice(-(scant[i]+1),None)
        tmpshape[dim] = slice(scant[i]+1)
        outshape[dim] = i
        out[tuple(outshape)]=torch.sum(filter[tuple(shape)]*tensor[tuple(tmpshape)],dim=dim)
    return out
def Myconvolve(tensor,filter,dim=0):
    dimsize = tensor.size(dim)
    shape = tensor.shape
    dims = torch.arange(len(shape)).tolist()
    dims[dim] = len(shape)-1
    dims[-1] = dim
    tensor = tensor.permute(*dims)
    after_shape = tensor.shape
    tensor = tensor.reshape(-1,1,dimsize)
    filter = torch.flip(filter.squeeze(),[0])
    assert dimsize==len(filter),f"error :filter shape:{filter.shape},dimsize={dimsize}"
    out = torch.conv1d(tensor,filter.view(1,1,-1),padding=len(filter)-1)
    out = out[...,:dimsize]
    out = out.view(*after_shape).permute(*dims)
    assert out.shape==shape,f"out.shape{out.shape},ori_shape{shape}"
    return out
def CpConvolveWithExp2Out(A,B,Cp,t,scant,decay,device=2):
    A = A.cuda(device)
    B = B.cuda(device)
    Cp = Cp.cuda(device).view(-1,1,1)
    t = t.cuda(device).view(-1,1,1)
    scant  =scant.cuda(device)
    tmp = A*(MyCumtrapz(Cp*torch.exp(-decay*t),x=t,dim=0,initial=0)-torch.exp(-decay*t)*(Myconvolve(torch.exp(-B*t),Cp,dim=0)-0.5*(Cp+Cp[0:1]*torch.exp(-B*t)))/60)/(B+decay)
    tmp = (tmp[scant[1:]]-tmp[scant[:-1]])/(scant[1:]-scant[:-1]).view(-1,1,1)*60
    return (tmp).cuda(3)
def CpConvolveWithExp2WithoutDevice(A,B,Cp,t,scant,decay,split_size = (128,128)):
    Cp = Cp.view(-1,1,1)
    t = t.view(-1,1,1)
    b,h,w = A.shape
    out = torch.zeros((len(scant)-1,h,w),device=A.device)
    for i in range(int(h/split_size[0]+0.5)):
        for j in range(int(w/split_size[1]+0.5)):
            tmp = A[:,i*split_size[0]:(i+1)*split_size[0],j*split_size[1]:(j+1)*split_size[1]]*((MySimpleconvolve(torch.exp(-B[:,i*split_size[0]:(i+1)*split_size[0],j*split_size[1]:(j+1)*split_size[1]]*t),Cp,dim=0,scant=scant[:-1])-0.5*(Cp+Cp[0:1]*torch.exp(-B[:,i*split_size[0]:(i+1)*split_size[0],j*split_size[1]:(j+1)*split_size[1]]*t))[scant[:-1]])/60)
            # tmp = tmp[scant[:-1]]
            out[:,i*split_size[0]:(i+1)*split_size[0],j*split_size[1]:(j+1)*split_size[1]]=tmp
    return out

def train_one_iter(loss_fn,batch,t_all,decay,scant,net,project_net,kparam_net):
    SUV,cp,ki,intercept = batch
    SUV = SUV.float().cuda()
    ki = ki.float().cuda()
    intercept = intercept.float().cuda()
    cp = cp.float().cuda().flatten()
    assert len(cp)==3901,"error"
    int_cp = MyCumtrapz(cp, t_all, initial=0)
    x = ((int_cp) / (cp))[scant[15:-1]]
    idx = 22
    SUV.squeeze_()
    input = SUV[:idx]
    T,h,w = input.shape
    out = net(input.view(T,1,h,w))
    t,c,h,w = out.shape
    out = out.view(1,-1,h,w)
    out = project_net(out)
    k_param = kparam_net(out)
    k_param = k_param.squeeze_()
    out = torch.concat([SUV[:idx],k_param],dim=0)
    assert len(out)==len(SUV),f"out.shape{out.shape} != SUV.shape{SUV.shape}"
    ki_fit_pred = out[15:]/cp[scant[15:-1]].view(-1,1,1)
    ki_fit_true = ki*x.view(-1,1,1)+intercept
    loss = 10*loss_fn(ki_fit_pred,ki_fit_true)+loss_fn(torch.diff(10*ki_fit_pred,dim=0),torch.diff(10*ki_fit_true,dim=0))

    return loss
def train_one_iter_val(loss_fn,batch,t_all,decay,scant,net,project_net,kparam_net):
    SUV, cp, ki, intercept = batch
    SUV = SUV.float().cuda()
    ki = ki.float().cuda()
    intercept = intercept.float().cuda()
    cp = cp.float().cuda().flatten()
    assert len(cp) == 3901, "error"
    int_cp = MyCumtrapz(cp, t_all, initial=0)
    x = ((int_cp) / (cp))[scant[15:-1]]
    idx = 22
    SUV.squeeze_()
    input = SUV[:idx]
    T,h,w = input.shape
    out = net(input.view(T,1,h,w))
    t,c,h,w = out.shape
    out = out.view(1,-1,h,w)
    out = project_net(out)
    k_param = kparam_net(out)
    k_param = k_param.squeeze_()
    out = torch.concat([SUV[:idx], k_param], dim=0)
    assert len(out) == len(SUV), f"out.shape{out.shape} != SUV.shape{SUV.shape}"
    ki_fit_pred = out[15:] / cp[scant[15:-1]].view(-1, 1, 1)
    ki_fit_true = ki * x.view(-1, 1, 1) + intercept
    loss = 10*loss_fn(ki_fit_pred, ki_fit_true) +  loss_fn(torch.diff(10*ki_fit_pred, dim=0),
                                                                                   torch.diff(10*ki_fit_true, dim=0))
    return loss,out,SUV
from torch.utils.data import Subset
def register_history(criterion):
    history = History()

    for k, v in criterion.items():
        name = k
        val_name = 'val_{}'.format(name)

        history.register(name, ':.4e')
        history.register(val_name, ':.4e')
    return history
def main(i,x,Y):
    y = Y[:,i]
    slope, intercept, r, p, se = linregress(x, y)
    return np.array([slope, intercept, r, p, se])
def compare_one_iter(loss_fn,batch,t_all,decay,scant,net,project_net,kparam_net):
    SUV,cp,ki,intercept = batch
    SUV = SUV.float().cuda()
    ki = ki.float().cuda()
    intercept = intercept.float().cuda()
    cp = cp.float().cuda().flatten()
    assert len(cp)==3901,"error"
    int_cp = MyCumtrapz(cp, t_all, initial=0)
    x = ((int_cp) / (cp))[scant[15:-1]]
    idx = 22
    SUV.squeeze_()
    input = SUV[:idx]
    T,h,w = input.shape
    out = net(input.view(T,1,h,w))
    t,c,h,w = out.shape
    out = out.view(1,-1,h,w)
    out = project_net(out)
    k_param = kparam_net(out)
    k_param = k_param.squeeze_()
    out = torch.concat([SUV[:idx],k_param],dim=0)
    assert len(out)==len(SUV),f"out.shape{out.shape} != SUV.shape{SUV.shape}"
    ki_fit_pred = out[15:]/cp[scant[15:-1]].view(-1,1,1)
    ki_fit_true = ki*x.view(-1,1,1)+intercept
    loss = 10*loss_fn(ki_fit_pred,ki_fit_true)+loss_fn(torch.diff(10*ki_fit_pred,dim=0),torch.diff(10*ki_fit_true,dim=0))

    return loss,out,SUV


if __name__ == '__main__':
    criterion = {"loss": nn.HuberLoss(delta=10)}
    history = register_history(criterion)
    net = Net()
    net = net.cuda()
    project_net = nn.Sequential(nn.Conv2d(220, 256, 1), nn.GroupNorm(256 // 16, 256), nn.LeakyReLU())
    kparam_net = nn.Sequential(nn.Conv2d(256, 128, 1), nn.GroupNorm(128 // 16, 128), nn.LeakyReLU(),
                               nn.Conv2d(128, 128, 1), nn.GroupNorm(128 // 16, 128), nn.LeakyReLU(),
                               nn.Conv2d(128, 64, 1), nn.GroupNorm(64 // 16, 64), nn.LeakyReLU(), nn.Conv2d(64, 5, 1))
    project_net = project_net.cuda().float()
    kparam_net = kparam_net.cuda().float()
    dataset = PETDataset()
    test_idx = [i for i in range(len(dataset) - 71 * 10, len(dataset))]
    test_dataset = Subset(dataset, test_idx)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    t_all = torch.arange(3901).float() / 60
    scant = torch.tensor((0, 10, 20, 30, 40, 50, 60, 90, 120, 150, 180, 240, 300, 360, 420, 540, 660, 780, 900, 1200,
                          1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900)).long()
    net.load_state_dict(torch.load("best_weights/net_version4_retrain.pkl"))
    project_net.load_state_dict(torch.load("best_weights/project_net_version4_retrain.pkl"))
    kparam_net.load_state_dict(torch.load("best_weights/kparam_net_version4_retrain.pkl"))
    main_path = r"./best_weights_version4_retain_result_all(final)"
    fsim_result_torch = np.zeros((len(test_dataset), 3))
    GMSD_result_torch = np.zeros((len(test_dataset), 3))
    VSI_result_torch = np.zeros((len(test_dataset), 3))
    DSS_result_torch = np.zeros((len(test_dataset), 3))
    MDSI_result_torch = np.zeros((len(test_dataset), 3))
    HaarPSI_result_torch = np.zeros((len(test_dataset), 3))
    iwssim_result_torch = np.zeros((len(test_dataset), 3))
    MSGMSD_result_torch = np.zeros((len(test_dataset), 3))
    msssim_result_torch = np.zeros((len(test_dataset), 3))
    PieAPP_result_torch = np.zeros((len(test_dataset), 3))
    psnr_result_torch = np.zeros((len(test_dataset), 3))
    srsim_result_torch = np.zeros((len(test_dataset), 3))
    ssim_result_torch = np.zeros((len(test_dataset), 3))
    psnr_result_ori = np.zeros((len(test_dataset), 3))
    ssim_result_ori = np.zeros((len(test_dataset), 3))
    rmse_result_ori = np.zeros((len(test_dataset), 3))
    nmi_result_ori = np.zeros((len(test_dataset), 3))
    fsim_result_torch_SUV = np.zeros((len(test_dataset), 2))
    GMSD_result_torch_SUV = np.zeros((len(test_dataset), 2))
    VSI_result_torch_SUV = np.zeros((len(test_dataset), 2))
    DSS_result_torch_SUV = np.zeros((len(test_dataset), 2))
    MDSI_result_torch_SUV = np.zeros((len(test_dataset), 2))
    HaarPSI_result_torch_SUV = np.zeros((len(test_dataset), 2))
    iwssim_result_torch_SUV = np.zeros((len(test_dataset), 2))
    MSGMSD_result_torch_SUV = np.zeros((len(test_dataset), 2))
    msssim_result_torch_SUV = np.zeros((len(test_dataset), 2))
    PieAPP_result_torch_SUV = np.zeros((len(test_dataset), 2))
    psnr_result_torch_SUV = np.zeros((len(test_dataset), 2))
    srsim_result_torch_SUV = np.zeros((len(test_dataset), 2))
    ssim_result_torch_SUV = np.zeros((len(test_dataset), 2))
    nmi_result_ori_SUV = np.zeros((len(test_dataset), 2))
    FR_metric = [pyiqa.create_metric(metric_name, device=torch.device("cuda"),as_loss=False) for metric_name in pyiqa.list_models(metric_mode='FR')]
    NR_metric = [pyiqa.create_metric(metric_name, device=torch.device("cuda"),as_loss=False) for metric_name in pyiqa.list_models(metric_mode='NR')
                 if metric_name != 'fid']
    fr_is_lower_better = list(map(lambda x: (x.metric_name, x.lower_better), FR_metric))
    nr_is_lower_better = list(map(lambda x: (x.metric_name, x.lower_better), NR_metric))
    result_fr_metric = [np.zeros((len(test_dataset), 2)) for metric_name in pyiqa.list_models(metric_mode='FR')]
    result_nr_metric = [np.zeros((len(test_dataset), 3)) for metric_name in pyiqa.list_models(metric_mode='NR')
                        if metric_name != 'fid']
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    with torch.no_grad():
        for i,batch in enumerate(dataloader):
            _, cp, _, _ = test_dataset[i]
            int_cp = MyCumtrapz(torch.from_numpy(cp), t_all, initial=0).detach().cpu().numpy()
            x = int_cp / cp
            x = x[scant[:-1].cpu().numpy()]
            x[0] = 0
            tmpi = i
            i=test_idx[i]
            if os.path.exists(f"{main_path}/{i}_pred.pkl") and os.path.exists(f"{main_path}/{i}_pred_compare.pkl"):
                pred = torch.load(f"{main_path}/{i}_pred.pkl")
                pred_compare = torch.load(f"{main_path}/{i}_pred_compare.pkl")
                gt = torch.load(f"{main_path}/{i}_gt.pkl")
            else:
                a = os.path.exists(f"{main_path}/{i}_pred.pkl")
                b = os.path.exists(f"{main_path}/{i}_pred_compare.pkl")
                raise Exception(f"error1:please run generate_predict_data.py first to generate {main_path}/{i}_pred.pkl (exist:{a}), {main_path}/{i}_pred_compare.pkl (exist:{b}).")
            if os.path.exists(f"{main_path}/{i}_Ki_ori_method.npy"):
                pred_Ki = np.load(f"{main_path}/{i}_Ki_pred.npy")
                pred_compare_Ki = np.load(f"{main_path}/{i}_Ki_pred_compare.npy")
                gt_Ki = np.load(f"{main_path}/{i}_Ki_gt.npy")
                ori_method_Ki = np.load(f"{main_path}/{i}_Ki_ori_method.npy")
            else:
                raise Exception("error2:please run generate_predict_data.py first to generate result data.")
            out_gt = gt_Ki[0].reshape(256,256)
            out_gt_SUV = gt[-1].reshape(256,256)
            out_pred_SUV = pred[-1].reshape(256,256)
            out_pred_compare_SUV = pred_compare[-1].reshape(256,256)
            out_ori_method_Ki = ori_method_Ki[0].reshape(256,256)
            out_pred_compare_Ki = pred_compare_Ki.reshape(256,256)
            out_pred_Ki = pred_Ki.reshape(256,256)
            out_gt = np.maximum(out_gt, 0)
            out_ori_method_Ki = np.maximum(out_ori_method_Ki, 0)
            out_pred_Ki = np.maximum(out_pred_Ki, 0)
            out_pred_compare_Ki = np.maximum(out_pred_compare_Ki, 0)
            out_gt_SUV = np.maximum(out_gt_SUV, 0)
            out_pred_SUV = np.maximum(out_pred_SUV, 0)
            out_pred_compare_SUV = np.maximum(out_pred_compare_SUV, 0)
            maxval = np.max(out_gt)
            maxval_SUV = np.max(out_gt_SUV)
            out_ori_method_Ki = np.minimum(out_ori_method_Ki,maxval)
            out_pred_Ki = np.minimum(out_pred_Ki,maxval)
            out_pred_compare_Ki = np.minimum(out_pred_compare_Ki,maxval)
            out_pred_SUV = np.minimum(out_pred_SUV,maxval_SUV)
            out_pred_compare_SUV = np.minimum(out_pred_compare_SUV,maxval_SUV)
            i = tmpi
            out_pred_Ki_torch = torch.from_numpy(out_pred_Ki).unsqueeze(0).unsqueeze(0).float()
            out_pred_compare_Ki_torch = torch.from_numpy(out_pred_compare_Ki).unsqueeze(0).unsqueeze(0).float()
            out_gt_torch = torch.from_numpy(out_gt).unsqueeze(0).unsqueeze(0).float()
            out_ori_method_Ki_torch = torch.from_numpy(out_ori_method_Ki).unsqueeze(0).unsqueeze(0).float()
            psnr_result_torch[i, 0] = piq.psnr(out_pred_Ki_torch, out_gt_torch, data_range=maxval).item()
            psnr_result_torch[i, 1] = piq.psnr(out_ori_method_Ki_torch, out_gt_torch, data_range=maxval).item()
            psnr_result_torch[i, 2] = piq.psnr(out_pred_compare_Ki_torch, out_gt_torch, data_range=maxval).item()
            ssim_result_torch[i, 0] = piq.ssim(out_pred_Ki_torch, out_gt_torch, data_range=maxval).item()
            ssim_result_torch[i, 1] = piq.ssim(out_ori_method_Ki_torch, out_gt_torch, data_range=maxval).item()
            ssim_result_torch[i, 2] = piq.ssim(out_pred_compare_Ki_torch, out_gt_torch, data_range=maxval).item()
            msssim_result_torch[i, 0] = piq.multi_scale_ssim(out_pred_Ki_torch, out_gt_torch, data_range=maxval).item()
            msssim_result_torch[i, 1] = piq.multi_scale_ssim(out_ori_method_Ki_torch, out_gt_torch, data_range=maxval).item()
            msssim_result_torch[i, 2] = piq.multi_scale_ssim(out_pred_compare_Ki_torch, out_gt_torch, data_range=maxval).item()
            iwssim_result_torch[i, 0] = piq.information_weighted_ssim(out_pred_Ki_torch, out_gt_torch, data_range=maxval).item()
            iwssim_result_torch[i, 1] = piq.information_weighted_ssim(out_ori_method_Ki_torch, out_gt_torch,
                                                                data_range=maxval).item()
            iwssim_result_torch[i, 2] = piq.information_weighted_ssim(out_pred_compare_Ki_torch, out_gt_torch,
                                                                data_range=maxval).item()
            fsim_result_torch[i, 0] = piq.fsim(out_pred_Ki_torch, out_gt_torch,
                                                                 data_range=maxval,chromatic=False).item()
            fsim_result_torch[i, 1] = piq.fsim(out_ori_method_Ki_torch, out_gt_torch,
                                                                 data_range=maxval,chromatic=False).item()
            fsim_result_torch[i, 2] = piq.fsim(out_pred_compare_Ki_torch, out_gt_torch,
                                                                    data_range=maxval,chromatic=False).item()
            srsim_result_torch[i, 0] = piq.srsim(out_pred_Ki_torch, out_gt_torch,
                                                                    data_range=maxval,chromatic=False).item()
            srsim_result_torch[i, 1] = piq.srsim(out_ori_method_Ki_torch, out_gt_torch,
                                                                    data_range=maxval,chromatic=False).item()
            srsim_result_torch[i, 2] = piq.srsim(out_pred_compare_Ki_torch, out_gt_torch,
                                                                    data_range=maxval,chromatic=False).item()
            GMSD_result_torch[i, 0] = piq.gmsd(out_pred_Ki_torch, out_gt_torch,
                                                                      data_range=maxval).item()
            GMSD_result_torch[i, 1] = piq.gmsd(out_ori_method_Ki_torch, out_gt_torch,
                                                                        data_range=maxval).item()
            GMSD_result_torch[i, 2] = piq.gmsd(out_pred_compare_Ki_torch, out_gt_torch,
                                                                        data_range=maxval).item()
            VSI_result_torch[i, 0] = piq.vsi(out_pred_Ki_torch, out_gt_torch,
                                                                    data_range=maxval).item()
            VSI_result_torch[i, 1] = piq.vsi(out_ori_method_Ki_torch, out_gt_torch,
                                                                    data_range=maxval).item()
            VSI_result_torch[i, 2] = piq.vsi(out_pred_compare_Ki_torch, out_gt_torch,
                                                                    data_range=maxval).item()
            DSS_result_torch[i, 0] = piq.dss(out_pred_Ki_torch, out_gt_torch,
                                                                      data_range=maxval).item()
            DSS_result_torch[i, 1] = piq.dss(out_ori_method_Ki_torch, out_gt_torch,
                                                                        data_range=maxval).item()
            DSS_result_torch[i, 2] = piq.dss(out_pred_compare_Ki_torch, out_gt_torch,
                                                                        data_range=maxval).item()
            HaarPSI_result_torch[i, 0] = piq.haarpsi(out_pred_Ki_torch, out_gt_torch,
                                                                      data_range=maxval).item()
            HaarPSI_result_torch[i, 1] = piq.haarpsi(out_ori_method_Ki_torch, out_gt_torch,
                                                                        data_range=maxval).item()
            HaarPSI_result_torch[i, 2] = piq.haarpsi(out_pred_compare_Ki_torch, out_gt_torch,
                                                                        data_range=maxval).item()
            MDSI_result_torch[i, 0] = piq.mdsi(out_pred_Ki_torch, out_gt_torch,
                                                                        data_range=maxval).item()
            MDSI_result_torch[i, 1] = piq.mdsi(out_ori_method_Ki_torch, out_gt_torch,
                                                                        data_range=maxval).item()
            MDSI_result_torch[i, 2] = piq.mdsi(out_pred_compare_Ki_torch, out_gt_torch,
                                                                        data_range=maxval).item()
            MSGMSD_result_torch[i, 0] = piq.multi_scale_gmsd(out_pred_Ki_torch, out_gt_torch,
                                                                         data_range=maxval,chromatic=False).item()
            MSGMSD_result_torch[i, 1] = piq.multi_scale_gmsd(out_ori_method_Ki_torch, out_gt_torch,
                                                                         data_range=maxval,chromatic=False).item()
            MSGMSD_result_torch[i, 2] = piq.multi_scale_gmsd(out_pred_compare_Ki_torch, out_gt_torch,
                                                                         data_range=maxval,chromatic=False).item()
            PieAPP_result_torch[i, 0] = piq.pieapp(out_pred_Ki_torch, out_gt_torch,
                                                                         data_range=maxval).item()
            ssim_result_ori[i,0] = ssim(out_gt,out_pred_Ki,data_range=maxval)
            ssim_result_ori[i,1] = ssim(out_gt,out_ori_method_Ki,data_range=maxval)
            ssim_result_ori[i,2] = ssim(out_gt,out_pred_compare_Ki,data_range=maxval)
            psnr_result_ori[i,0] = psnr(out_gt,out_pred_Ki,data_range=maxval)
            psnr_result_ori[i,1] = psnr(out_gt,out_ori_method_Ki,data_range=maxval)
            psnr_result_ori[i,2] = psnr(out_gt,out_pred_compare_Ki,data_range=maxval)
            rmse_result_ori[i,0] = rmse(out_gt,out_pred_Ki)
            rmse_result_ori[i,1] = rmse(out_gt,out_ori_method_Ki)
            rmse_result_ori[i,2] = rmse(out_gt,out_pred_compare_Ki)
            nmi_result_ori[i,0] = nmi(out_gt,out_pred_Ki)
            nmi_result_ori[i,1] = nmi(out_gt,out_ori_method_Ki)
            nmi_result_ori[i,2] = nmi(out_gt,out_pred_compare_Ki)
            psnr_result_torch_SUV[i, 0] = piq.psnr(out_pred_compare_SUV, out_gt_torch, data_range=maxval_SUV).item()
            psnr_result_torch_SUV[i, 1] = piq.psnr(out_pred_SUV, out_gt_torch, data_range=maxval_SUV).item()
            ssim_result_torch_SUV[i, 0] = piq.ssim(out_pred_compare_SUV, out_gt_torch, data_range=maxval_SUV).item()
            ssim_result_torch_SUV[i, 1] = piq.ssim(out_pred_SUV, out_gt_torch, data_range=maxval_SUV).item()
            msssim_result_torch_SUV[i, 0] = piq.multi_scale_ssim(out_pred_compare_SUV, out_gt_torch, data_range=maxval_SUV).item()
            msssim_result_torch_SUV[i, 1] = piq.multi_scale_ssim(out_pred_SUV, out_gt_torch, data_range=maxval_SUV).item()
            iwssim_result_torch_SUV[i, 0] = piq.information_weighted_ssim(out_pred_compare_SUV, out_gt_torch, data_range=maxval_SUV).item()
            iwssim_result_torch_SUV[i, 1] = piq.information_weighted_ssim(out_pred_SUV, out_gt_torch,
                                                                    data_range=maxval_SUV).item()
            fsim_result_torch_SUV[i, 0] = piq.fsim(out_pred_compare_SUV, out_gt_torch,
                                                                 data_range=maxval_SUV,chromatic=False).item()
            fsim_result_torch_SUV[i, 1] = piq.fsim(out_pred_SUV, out_gt_torch,
                                                                 data_range=maxval_SUV,chromatic=False).item()
            srsim_result_torch_SUV[i, 0] = piq.srsim(out_pred_compare_SUV, out_gt_torch,
                                                                 data_range=maxval_SUV,chromatic=False).item()
            srsim_result_torch_SUV[i, 1] = piq.srsim(out_pred_SUV, out_gt_torch,
                                                                 data_range=maxval_SUV,chromatic=False).item()
            GMSD_result_torch_SUV[i, 0] = piq.gmsd(out_pred_compare_SUV, out_gt_torch,
                                                                    data_range=maxval_SUV).item()
            GMSD_result_torch_SUV[i, 1] = piq.gmsd(out_pred_SUV, out_gt_torch,
                                                                        data_range=maxval_SUV).item()
            VSI_result_torch_SUV[i, 0] = piq.vsi(out_pred_compare_SUV, out_gt_torch,
                                                                    data_range=maxval_SUV).item()
            VSI_result_torch_SUV[i, 1] = piq.vsi(out_pred_SUV, out_gt_torch,
                                                                    data_range=maxval_SUV).item()
            DSS_result_torch_SUV[i, 0] = piq.dss(out_pred_compare_SUV, out_gt_torch,
                                                                      data_range=maxval_SUV).item()
            DSS_result_torch_SUV[i, 1] = piq.dss(out_pred_SUV, out_gt_torch,
                                                                        data_range=maxval_SUV).item()
            HaarPSI_result_torch_SUV[i, 0] = piq.haarpsi(out_pred_compare_SUV, out_gt_torch,
                                                                      data_range=maxval_SUV).item()
            HaarPSI_result_torch_SUV[i, 1] = piq.haarpsi(out_pred_SUV, out_gt_torch,
                                                                        data_range=maxval_SUV).item()
            MDSI_result_torch_SUV[i, 0] = piq.mdsi(out_pred_compare_SUV, out_gt_torch,
                                                                        data_range=maxval_SUV).item()
            MDSI_result_torch_SUV[i, 1] = piq.mdsi(out_pred_SUV, out_gt_torch,
                                                                        data_range=maxval_SUV).item()
            MSGMSD_result_torch_SUV[i, 0] = piq.multi_scale_gmsd(out_pred_compare_SUV, out_gt_torch,
                                                                         data_range=maxval_SUV,chromatic=False).item()
            MSGMSD_result_torch_SUV[i, 1] = piq.multi_scale_gmsd(out_pred_SUV, out_gt_torch,
                                                                         data_range=maxval_SUV,chromatic=False).item()
            PieAPP_result_torch_SUV[i, 0] = piq.pieapp(out_pred_compare_SUV, out_gt_torch,
                                                                         data_range=maxval_SUV).item()
            PieAPP_result_torch_SUV[i, 1] = piq.pieapp(out_pred_SUV, out_gt_torch,
                                                                         data_range=maxval_SUV).item()
            nmi_result_ori_SUV[i,0] = nmi(out_gt_SUV.numpy(),out_pred_compare_SUV.numpy(),data_range=maxval_SUV)
            nmi_result_ori_SUV[i,1] = nmi(out_gt_SUV.numpy(),out_pred_SUV.numpy(),data_range=maxval_SUV)
            for metric, (name, is_lower),array_result in zip(FR_metric, fr_is_lower_better,result_fr_metric):
                array_result[i,0] = metric(out_pred_compare_SUV, out_gt_SUV).item()
                array_result[i, 1] = metric(out_pred_SUV, out_gt_SUV).item()
                print(i,name, f"is_lower_better:{is_lower}", f"pred:{array_result[i,0]},original:{array_result[i, 1]}")
            for metric, (name, is_lower),array_result in zip(NR_metric, nr_is_lower_better,result_nr_metric):
                array_result[i, 0] = metric(out_pred_compare_SUV).item()
                array_result[i, 1] = metric(out_pred_SUV).item()
                array_result[i, 2] = metric(out_gt_SUV).item()
                print(i,name, f"is_lower_better:{is_lower}", f"pred:{array_result[i, 0]},original:{array_result[i, 1]},gt:{array_result[i, 2]}")
            print(f"{i}",f"SSIM:{ssim_result_ori[i]},PSNR:{psnr_result_ori[i]},RMSE:{rmse_result_ori[i]},NMI:{nmi_result_ori[i]}")
            print(i,f"ssim:{ssim_result_torch[i]},psnr:{psnr_result_torch[i]},"
                    f"msssim:{msssim_result_torch[i]},iwssim:{iwssim_result_torch[i]},fsim:{fsim_result_torch[i]},srsim:{srsim_result_torch[i]},"
                    f"GMSD:{GMSD_result_torch[i]},VSI:{VSI_result_torch[i]},DSS:{DSS_result_torch[i]},HaarPSI:{HaarPSI_result_torch[i]},"
                    f"MDSI:{MDSI_result_torch[i]},MSGMSD:{MSGMSD_result_torch[i]},PieAPP:{PieAPP_result_torch[i]}")
        #之前都是直接修改代码的，现在整理后为了与原先的代码保存的文件保持一致，所以写法有些奇怪。
        np.save(f"{main_path}/ssim_result_ori_compare_Ki.npy",ssim_result_ori[:,[2,0]])
        np.save(f"{main_path}/psnr_result_ori_compare_Ki.npy",psnr_result_ori[:,[2,0]])
        np.save(f"{main_path}/rmse_result_ori_compare_Ki.npy",rmse_result_ori[:,[2,0]])
        np.save(f"{main_path}/nmi_result_ori_compare_Ki.npy",nmi_result_ori[:,[2,0]])
        np.save(f"{main_path}/ssim_result_torch_compare_Ki.npy", ssim_result_torch[:,[2,0]])
        np.save(f"{main_path}/psnr_result_torch_compare_Ki.npy", psnr_result_torch[:,[2,0]])
        np.save(f"{main_path}/msssim_result_torch_compare_Ki.npy", msssim_result_torch[:,[2,0]])
        np.save(f"{main_path}/iwssim_result_torch_compare_Ki.npy", iwssim_result_torch[:,[2,0]])
        np.save(f"{main_path}/fsim_result_torch_compare_Ki.npy", fsim_result_torch[:,[2,0]])
        np.save(f"{main_path}/srsim_result_torch_compare_Ki.npy", srsim_result_torch[:,[2,0]])
        np.save(f"{main_path}/GMSD_result_torch_compare_Ki.npy", GMSD_result_torch[:,[2,0]])
        np.save(f"{main_path}/VSI_result_torch_compare_Ki.npy", VSI_result_torch[:,[2,0]])
        np.save(f"{main_path}/DSS_result_torch_compare_Ki.npy", DSS_result_torch[:,[2,0]])
        np.save(f"{main_path}/HaarPSI_result_torch_compare_Ki.npy", HaarPSI_result_torch[:,[2,0]])
        np.save(f"{main_path}/MDSI_result_torch_compare_Ki.npy", MDSI_result_torch[:,[2,0]])
        np.save(f"{main_path}/MSGMSD_result_torch_compare_Ki.npy", MSGMSD_result_torch[:,[2,0]])
        np.save(f"{main_path}/PieAPP_result_torch_compare_Ki.npy", PieAPP_result_torch[:,[2,0]])

        np.save(f"{main_path}/ssim_result_ori.npy",ssim_result_ori[:,[0,1]])
        np.save(f"{main_path}/psnr_result_ori.npy",psnr_result_ori[:,[0,1]])
        np.save(f"{main_path}/rmse_result_ori.npy",rmse_result_ori[:,[0,1]])
        np.save(f"{main_path}/nmi_result_ori.npy",nmi_result_ori[:,[0,1]])
        np.save(f"{main_path}/ssim_result_torch.npy", ssim_result_torch[:,[0,1]])
        np.save(f"{main_path}/psnr_result_torch.npy", psnr_result_torch[:,[0,1]])
        np.save(f"{main_path}/msssim_result_torch.npy", msssim_result_torch[:,[0,1]])
        np.save(f"{main_path}/iwssim_result_torch.npy", iwssim_result_torch[:,[0,1]])
        np.save(f"{main_path}/fsim_result_torch.npy", fsim_result_torch[:,[0,1]])
        np.save(f"{main_path}/srsim_result_torch.npy", srsim_result_torch[:,[0,1]])
        np.save(f"{main_path}/GMSD_result_torch.npy", GMSD_result_torch[:,[0,1]])
        np.save(f"{main_path}/VSI_result_torch.npy", VSI_result_torch[:,[0,1]])
        np.save(f"{main_path}/DSS_result_torch.npy", DSS_result_torch[:,[0,1]])
        np.save(f"{main_path}/HaarPSI_result_torch.npy", HaarPSI_result_torch[:,[0,1]])
        np.save(f"{main_path}/MDSI_result_torch.npy", MDSI_result_torch[:,[0,1]])
        np.save(f"{main_path}/MSGMSD_result_torch.npy", MSGMSD_result_torch[:,[0,1]])
        np.save(f"{main_path}/PieAPP_result_torch.npy", PieAPP_result_torch[:,[0,1]])

        np.save(f"{main_path}/nmi_result_ori_compare_SUV.npy",nmi_result_ori_SUV)
        np.save(f"{main_path}/ssim_result_torch_compare_SUV.npy", ssim_result_torch_SUV)
        np.save(f"{main_path}/psnr_result_torch_compare_SUV.npy", psnr_result_torch_SUV)
        np.save(f"{main_path}/msssim_result_torch_compare_SUV.npy", msssim_result_torch_SUV)
        np.save(f"{main_path}/iwssim_result_torch_compare_SUV.npy", iwssim_result_torch_SUV)
        np.save(f"{main_path}/fsim_result_torch_compare_SUV.npy", fsim_result_torch_SUV)
        np.save(f"{main_path}/srsim_result_torch_compare_SUV.npy", srsim_result_torch_SUV)
        np.save(f"{main_path}/GMSD_result_torch_compare_SUV.npy", GMSD_result_torch_SUV)
        np.save(f"{main_path}/VSI_result_torch_compare_SUV.npy", VSI_result_torch_SUV)
        np.save(f"{main_path}/DSS_result_torch_compare_SUV.npy", DSS_result_torch_SUV)
        np.save(f"{main_path}/HaarPSI_result_torch_compare_SUV.npy", HaarPSI_result_torch_SUV)
        np.save(f"{main_path}/MDSI_result_torch_compare_SUV.npy", MDSI_result_torch_SUV)
        np.save(f"{main_path}/MSGMSD_result_torch_compare_SUV.npy", MSGMSD_result_torch_SUV)
        np.save(f"{main_path}/PieAPP_result_torch_compare_SUV.npy", PieAPP_result_torch_SUV)

        np.save(f"{main_path}/result_fr_metric_compare_SUV.npy",result_fr_metric)
        np.save(f"{main_path}/result_nr_metric_compare_SUV.npy", result_nr_metric)
        np.save(f"{main_path}/fr_is_lower_better_compare_SUV.npy", fr_is_lower_better)
        np.save(f"{main_path}/nr_is_lower_better_compare_SUV.npy", nr_is_lower_better)

