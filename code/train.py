import traceback
import pydicom
import numpy as np
import os
from unet.unet_model import UNet
import torch.nn as nn
import torch
import torch.nn.functional as F
import re
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
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
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
    # with torch.no_grad():
    #     input = input.cuda()
    #     input_mean = torch.mean(input,dim=0,keepdim=True)
    #     input_std = torch.std(input,dim=0,keepdim=True)
    #     input = (input-input_mean)/(input_std+1e-20)
    out = net(input.view(T,1,h,w))
    t,c,h,w = out.shape
    out = out.view(1,-1,h,w)
    out = project_net(out)
    k_param = kparam_net(out)
    assert k_param.shape[0]==1,f"k_param shape error, get{k_param.shape}"
    fv = k_param[:,0,:,:]
    k1 = k_param[:,1,:,:]
    k2 = k_param[:,2,:,:]
    k3 = k_param[:,3,:,:]
    k4 = k_param[:,4,:,:]
    delta = torch.sqrt((k2+k3+k4)**2-4*k2*k4+1e-20)-1e-10
    # print("delta.isnan",torch.any(torch.isnan(delta)))
    b = torch.zeros_like(delta)
    b[delta==0] = k1[delta==0]/2
    b[delta!=0] = k1[delta!=0] / (2 * delta[delta!=0]) * ((k2 - k3 - k4 + delta)[delta!=0])
    # print("a.isnan",torch.any(torch.isnan(a)))
    a = torch.zeros_like(delta)
    a[delta == 0] = k1[delta == 0] / 2
    a[delta != 0] = k1[delta != 0] / (2 * delta[delta != 0]) * ((-k2 + k3 + k4 + delta)[delta != 0])
    # print("b.isnan",torch.any(torch.isnan(b)))
    c = 0.5 * (k2 + k3 + k4 - delta)
    # print("c.isnan",torch.any(torch.isnan(c)))
    d = 0.5 * (k2 + k3 + k4 + delta)
    # print("d.isnan",torch.any(torch.isnan(d)))
    afterConvolve1 = CpConvolveWithExp2WithoutDevice(a,c,cp,t_all,scant,decay)
    afterConvolve2 = CpConvolveWithExp2WithoutDevice(b,d,cp,t_all,scant,decay)
    afterConvolve  = afterConvolve1+afterConvolve2
    # print("afterConvolve.isnan",torch.any(torch.isnan(afterConvolve)))
    whole_blood_integrate = cp[scant[:-1]].view(-1,1,1)
    # print("whole_blood_integrate.isnan",torch.any(torch.isnan(whole_blood_integrate)))
    out = (1-fv)*afterConvolve+fv*whole_blood_integrate
    del afterConvolve
    del whole_blood_integrate
    del  fv
    ki_fit_pred = out[15:]/cp[scant[15:-1]].view(-1,1,1)
    ki_fit_true = ki*x.view(-1,1,1)+intercept
    loss = 10*loss_fn(ki_fit_pred,ki_fit_true)+loss_fn(torch.diff(10*ki_fit_pred,dim=0),torch.diff(10*ki_fit_true,dim=0))

    return loss
def train_net(loss_name,net,project_net,kparam_net,dataloader,val_dataloader,criterion,history:History,decay=float(np.log(2))/109.8,lr=1e-4,epochs=200,iter_output=100,split_image_size=(16,16),retrain=False,retrain_lr=False,load_path=None):
    net = net.float().cuda()
    project_net = project_net.float().cuda()
    loss_fn = criterion[loss_name]
    optimizer_net = optim.RMSprop(net.parameters(), lr=lr)
    optimizer_project_net = optim.RMSprop(project_net.parameters(), lr=lr)
    optimizer_kparam_net = optim.RMSprop(kparam_net.parameters(), lr=lr)
    minval = float("inf")
    writer = SummaryWriter(log_dir="log_version4_retrain", filename_suffix=str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    if retrain:
        if os.path.exists("best_weights/net_version4_retrain.pkl"):
            net.load_state_dict(torch.load("best_weights/net_version4_retrain.pkl"))
            project_net.load_state_dict(torch.load("best_weights/project_net_version4_retrain.pkl"))
            kparam_net.load_state_dict(torch.load("best_weights/kparam_net_version4_retrain.pkl"))
            optimizer_net.load_state_dict(torch.load("best_weights/optimizer_net_version4_retrain.pkl"))
            optimizer_project_net.load_state_dict(torch.load("best_weights/optimizer_project_net_version4_retrain.pkl"))
            optimizer_kparam_net.load_state_dict(torch.load("best_weights/optimizer_kparam_net_version4_retrain.pkl"))
            minval = torch.load("best_weights/minval_version4_retrain.pkl")
            print(f"resume model, minval is {minval}")
    else:
        if load_path is not None:
            net.load_state_dict(torch.load(load_path["net"]))
            project_net.load_state_dict(torch.load(load_path["project_net"]))
            kparam_net.load_state_dict(torch.load(load_path["kparam_net"]))
    if retrain_lr:
        for param_group in chain(optimizer_kparam_net.param_groups,optimizer_project_net.param_groups):
            param_group['lr']=lr
        for param_group in optimizer_net.param_groups:
            param_group['lr'] = 1e-7
    # lr_scheduler_net = MultiStepLR(optimizer_net, milestones=[8000,16000,24000])
    lr_scheduler_project_net = ReduceLROnPlateau(optimizer_project_net,mode='min', factor=0.1, patience=5500, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)
    lr_scheduler_kparam_net = ReduceLROnPlateau(optimizer_kparam_net,mode='min', factor=0.1, patience=5500, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)
    global_step = 0
    val_step = 0
    loss_record = EMA(1000)
    train_minval = float("inf")
    t_all = torch.arange(3901).float().cuda()/60
    scant = torch.tensor((0,10,20,30,40,50,60,90,120,150,180,240,300,360,420,540,660,780,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,3900)).long().cuda()
    for epoch in range(epochs):
        net.train()
        project_net.train()
        kparam_net.train()
        s_time = time.time()
        history.reset()
        for it,batch in enumerate(dataloader):
            # with autograd.detect_anomaly():
            loss = train_one_iter(loss_fn,batch,t_all,decay,scant,net,project_net,kparam_net)
            if torch.all(torch.isnan(loss)):
                print("loss is nan error!")
                torch.save(batch,f"error/epoch{epoch}_it{it}.pkl")
                continue
            optimizer_net.zero_grad(set_to_none=True)
            optimizer_project_net.zero_grad(set_to_none=True)
            optimizer_kparam_net.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(chain(net.parameters(),project_net.parameters(),kparam_net.parameters()), 100000, norm_type=float("inf"),error_if_nonfinite=True)
            if norm>100000:
                print(f"epoch{epoch},it{it},norm:",norm)
            optimizer_net.step()
            optimizer_project_net.step()
            optimizer_kparam_net.step()
            loss_record.update(loss.item())
            if it>1000 and it%500==0 and train_minval>loss_record.val():
                train_minval = loss_record.val()
                torch.save(net.state_dict(),f"train_process_weights/net_version4_retrain_epoch{epoch}_it{it}_loss{train_minval}.pkl")
                torch.save(project_net.state_dict(),f"train_process_weights/project_net_version4_retrain_epoch{epoch}_it{it}_loss{train_minval}.pkl")
                torch.save(kparam_net.state_dict(),f"train_process_weights/kparam_net_version4_retrain_epoch{epoch}_it{it}_loss{train_minval}.pkl")
            history.update(1,loss_name,loss.item())
            history.update(n=1, name='BatchTime', val=time.time() - s_time)
            writer.add_scalar("loss_train",loss.item(),global_step=global_step)
            global_step+=1
            if it%iter_output==0:
                history.display(epoch,it,len(dataloader),mode="train")
            s_time = time.time()
            del loss
            torch.cuda.empty_cache()
            # lr_scheduler_net.step()
            lr_scheduler_project_net.step(loss_record.val())
            lr_scheduler_kparam_net.step(loss_record.val())
        net.eval()
        project_net.eval()
        kparam_net.eval()
        with torch.no_grad():
            for i,batch in enumerate(val_dataloader):
                loss = train_one_iter(loss_fn,batch,t_all,decay,scant,net,project_net,kparam_net)
                history.update(1,"val_"+loss_name,loss.item())
                writer.add_scalar("loss_val",loss.item(),global_step=val_step)
                val_step+=1
                del loss
                torch.cuda.empty_cache()
            history.display(epoch,mode="validate")

        if minval>history.history_dict["val_"+loss_name].avg:
            torch.save(net.state_dict(),"best_weights/net_version4_retrain.pkl")
            torch.save(project_net.state_dict(),"best_weights/project_net_version4_retrain.pkl")
            torch.save(kparam_net.state_dict(),"best_weights/kparam_net_version4_retrain.pkl")
            torch.save(optimizer_net.state_dict(),"best_weights/optimizer_net_version4_retrain.pkl")
            torch.save(optimizer_project_net.state_dict(),"best_weights/optimizer_project_net_version4_retrain.pkl")
            torch.save(optimizer_kparam_net.state_dict(),"best_weights/optimizer_kparam_net_version4_retrain.pkl")
            minval = history.history_dict["val_"+loss_name].avg
            torch.save(minval,"best_weights/minval_version4_retrain.pkl")
    return net,project_net,kparam_net
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
    assert k_param.shape[0]==1,f"k_param shape error, get{k_param.shape}"
    fv = k_param[:,0,:,:]
    k1 = k_param[:,1,:,:]
    k2 = k_param[:,2,:,:]
    k3 = k_param[:,3,:,:]
    k4 = k_param[:,4,:,:]
    delta = torch.sqrt((k2+k3+k4)**2-4*k2*k4+1e-20)
    # print("delta.isnan",torch.any(torch.isnan(delta)))
    b = k1 / (2 * delta) * (k2 - k3 - k4 + delta)
    # print("a.isnan",torch.any(torch.isnan(a)))
    a = k1 / (2 * delta) * (-k2 + k3 + k4 + delta)
    # print("b.isnan",torch.any(torch.isnan(b)))
    c = 0.5 * (k2 + k3 + k4 - delta)
    # print("c.isnan",torch.any(torch.isnan(c)))
    d = 0.5 * (k2 + k3 + k4 + delta)
    # print("d.isnan",torch.any(torch.isnan(d)))
    afterConvolve1 = CpConvolveWithExp2WithoutDevice(a, c, cp, t_all, scant, decay)
    afterConvolve2 = CpConvolveWithExp2WithoutDevice(b, d, cp, t_all, scant, decay)
    afterConvolve = afterConvolve1 + afterConvolve2
    # print("afterConvolve.isnan",torch.any(torch.isnan(afterConvolve)))
    whole_blood_integrate = cp[scant[:-1]].view(-1, 1, 1)
    # print("whole_blood_integrate.isnan",torch.any(torch.isnan(whole_blood_integrate)))
    out = (1 - fv) * afterConvolve + fv * whole_blood_integrate
    del afterConvolve
    del whole_blood_integrate
    del  fv
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
if __name__=="__main__":
    criterion = {"loss":nn.HuberLoss(delta=20)}
    history = register_history(criterion)
    dataset = PETDataset()
    net = Net()
    net =net.cuda()
    train_idx = [i for i in range(len(dataset)-71*20) if i//71!=40 ]
    val_idx = [i for i in range(len(dataset)-71*20,len(dataset)-71*10) if i//71!=40 ]
    project_net = nn.Sequential(nn.Conv2d(220,256,1),nn.GroupNorm(256//16,256),nn.LeakyReLU())
    kparam_net = nn.Sequential(nn.Conv2d(256,128,1),nn.GroupNorm(128//16,128),nn.LeakyReLU(),nn.Conv2d(128,128,1),nn.GroupNorm(128//16,128),nn.LeakyReLU(),nn.Conv2d(128,64,1),nn.GroupNorm(64//16,64),nn.LeakyReLU(),nn.Conv2d(64,5,1),nn.Hardsigmoid())
    project_net = project_net.cuda().float()
    kparam_net = kparam_net.cuda().float()

    train_net("loss",net,project_net,kparam_net,DataLoader(Subset(dataset,train_idx),batch_size=1,shuffle=True),DataLoader(Subset(dataset,val_idx),batch_size=1,shuffle=False),criterion,history,lr=0.5e-4,iter_output=1,load_path={"net":"best_weights/net_version4_retrain.pkl","project_net":"best_weights/project_net_version4_retrain.pkl","kparam_net":"best_weights/kparam_net_version4_retrain.pkl"})