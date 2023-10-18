import traceback
import pydicom
import os
from unet.unet_model import UNet
import torch.nn as nn
from scipy.stats import linregress
import pandas as pd
import numpy as np

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
    is_compare_method = False
    criterion = {"loss": nn.HuberLoss(delta=10)}
    history = register_history(criterion)
    net = Net()
    net = net.cuda()
    project_net = nn.Sequential(nn.Conv2d(220, 256, 1), nn.GroupNorm(256 // 16, 256), nn.LeakyReLU())
    kparam_net = nn.Sequential(nn.Conv2d(256, 128, 1), nn.GroupNorm(128 // 16, 128), nn.LeakyReLU(),
                               nn.Conv2d(128, 128, 1), nn.GroupNorm(128 // 16, 128), nn.LeakyReLU(),
                               nn.Conv2d(128, 64, 1), nn.GroupNorm(64 // 16, 64), nn.LeakyReLU(), nn.Conv2d(64, 7, 1) if is_compare_method else nn.Conv2d(64, 5, 1),nn.Identity() if is_compare_method else nn.Hardsigmoid())
    project_net = project_net.cuda().float()
    kparam_net = kparam_net.cuda().float()
    dataset = PETDataset()
    test_idx = [i for i in range(len(dataset) - 71 * 10, len(dataset))]
    test_dataset = Subset(dataset, test_idx)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    t_all = torch.arange(3901).float() / 60
    scant = torch.tensor((0, 10, 20, 30, 40, 50, 60, 90, 120, 150, 180, 240, 300, 360, 420, 540, 660, 780, 900, 1200,
                          1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900)).long()
    postfix = "_compare" if is_compare_method else ""
    net.load_state_dict(torch.load(f"best_weights/net_version4_{'compare' if is_compare_method else 'retain'}.pkl"))
    project_net.load_state_dict(torch.load(f"best_weights/project_net_version4_{'compare' if is_compare_method else 'retain'}.pkl"))
    kparam_net.load_state_dict(torch.load(f"best_weights/kparam_net_version4_{'compare' if is_compare_method else 'retain'}.pkl"))
    main_path = r"./best_weights_version4_retain_result_all(final)"
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    with torch.no_grad():
        for i,batch in enumerate(dataloader):
            _, cp, _, _ = test_dataset[i]
            int_cp = MyCumtrapz(torch.from_numpy(cp), t_all, initial=0).detach().cpu().numpy()
            x = int_cp / cp
            x = x[scant[:-1].cpu().numpy()]
            x[0] = 0
            def func(vals):
                return linregress(x, vals)
            if os.path.exists(f"{main_path}/{i}_pred{postfix}.pkl") and os.path.exists(f"{main_path}/{i}_gt.pkl"):
                pass
            else:
                loss,pred,gt = train_one_iter_val(nn.HuberLoss(delta=30),batch,t_all.cuda(),float(np.log(2)/109.8),scant.cuda(),net,project_net,kparam_net)
                torch.save(pred,f"{main_path}/{i}_pred{postfix}.pkl")
                torch.save(gt,f"{main_path}/{i}_gt.pkl")
            if os.path.exists(f"{main_path}/{i}_Ki_pred{postfix}.npy") and os.path.exists(f"{main_path}/{i}_Ki_gt.npy") and os.path.exists(f"{main_path}/{i}_Ki_ori_method.npy"):
                pass
            else:
                _,cp,_,_ = batch
                cp = cp.cuda().flatten()
                pred_y = ((pred.cpu())/(cp[scant[:-1]].view(28,1,1).cpu())).detach().cpu().numpy().reshape(28,256*256)[22:]
                gt_y = ((gt.cpu()) / (cp[scant[:-1]].view(28, 1, 1).cpu())).detach().cpu().numpy().reshape(28,
                                                                                                            256 * 256)[15:]
                ori_y = ((gt.cpu()) / (cp[scant[:-1]].view(28, 1, 1).cpu())).detach().cpu().numpy().reshape(28,
                                                                                                            256 * 256)[15:22]
                cp = cp.cpu().numpy()
                # pred_y[0] = 0
                df = pd.DataFrame(pred_y)

                # pred_Ki = df.parallel_apply(func, result_type='expand').rename(index={0: 'slope', 1:'intercept', 2: 'rvalue', 3:'p-value', 4:'stderr'})
                pred_Ki = df.apply(lambda vals:linregress(x[22:], vals), result_type='expand').rename(
                    index={0: 'slope', 1: 'intercept', 2: 'rvalue', 3: 'p-value', 4: 'stderr'})

                pred_Ki = pred_Ki.to_numpy()
                np.save(f"{main_path}/{i}_Ki_pred{postfix}.npy",pred_Ki)
                df = pd.DataFrame(gt_y)

                gt_Ki = df.apply(lambda vals:linregress(x[15:], vals), result_type='expand').rename(
                    index={0: 'slope', 1: 'intercept', 2: 'rvalue', 3: 'p-value', 4: 'stderr'})

                gt_Ki = gt_Ki.to_numpy()
                np.save(f"{main_path}/{i}_Ki_gt.npy", gt_Ki)
                df = pd.DataFrame(ori_y)

                ori_method_Ki = df.apply(lambda vals:linregress(x[15:22], vals), result_type='expand').rename(
                    index={0: 'slope', 1: 'intercept', 2: 'rvalue', 3: 'p-value', 4: 'stderr'})

                ori_method_Ki = ori_method_Ki.to_numpy()
                np.save(f"{main_path}/{i}_Ki_ori_method.npy", ori_method_Ki)