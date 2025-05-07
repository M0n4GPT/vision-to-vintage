'''  
train_style_transfer.py  

Model Structure:
- Encoder: Full Pretrained VGG19 (all convolutional layers, frozen)
- AdaIN: Adaptive Instance Normalization aligning content/style features
- Decoder: Lightweight CNN with Upsample + Conv layers to reconstruct image

Usage examples (run on 4×A100 servers via torchrun):


# Single GPU
python train_style_transfer.py \
    --data_root "$IMG_DATA_DIR" \
    --global_batch_size 32 \
    --micro_batch_size 8 \
    --epochs 5 \
    --precision fp32 \
    --strategy none \
    --export_path ./stylizer.pt

# 4-GPU DDP
torchrun --nproc_per_node=4 train_style_transfer.py \
    --data_root /home/cc/img-dataset \
    --global_batch_size 128 \
    --micro_batch_size 8 \
    --epochs 5 \
    --precision amp \
    --strategy ddp \
    --export_path ./stylizer_ddp.pt

# 4-GPU FSDP
torchrun --nproc_per_node=4 train_style_transfer.py \
    --data_root /home/cc/img-dataset \
    --global_batch_size 128 \
    --micro_batch_size 8 \
    --epochs 5 \
    --precision amp \
    --strategy fsdp \
    --export_path ./stylizer_fsdp.pt
'''

import os
import time
import argparse
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models
from PIL import Image
import glob
import random

import subprocess
import mlflow
import mlflow.pytorch

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import CPUOffload, ShardingStrategy
except ImportError:
    FSDP = None


### Configure MLFlow
mlflow.set_tracking_uri("http://129.114.108.92:8000/") 
mlflow.set_experiment("style-trans")

# ---------- Dataset ----------
class StyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_root, transform):
        self.content_paths = glob.glob(os.path.join(content_dir, '*'))
        self.style_paths = []  # list of (path, label)
        style_dirs = sorted(os.listdir(style_root))
        for label in style_dirs:
            cls_dir = os.path.join(style_root, label)
            if os.path.isdir(cls_dir):
                for p in glob.glob(os.path.join(cls_dir, '*')):
                    self.style_paths.append((p, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, idx):
        c_img = Image.open(self.content_paths[idx]).convert('RGB')
        s_path, label = random.choice(self.style_paths)
        s_img = Image.open(s_path).convert('RGB')
        return self.transform(c_img), self.transform(s_img), label

# ---------- AdaIN ----------
def adain(content_feat: Tensor, style_feat: Tensor, eps: float = 1e-5) -> Tensor:
    c_mean = content_feat.mean(dim=[2,3], keepdim=True)
    c_std  = content_feat.std(dim=[2,3], keepdim=True)
    s_mean = style_feat.mean(dim=[2,3], keepdim=True)
    s_std  = style_feat.std(dim=[2,3], keepdim=True)
    norm = (content_feat - c_mean) / (c_std + eps)
    return norm * s_std + s_mean

# ---------- Model ----------
class StyleTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.enc = nn.Sequential(*list(vgg.children()))
        for p in self.enc.parameters(): p.requires_grad = False

        self.dec = nn.Sequential(
            nn.Conv2d(512,256,3,padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(256,128,3,padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(128,64,3,padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2,mode='nearest'),
            # 8→16→32→64→128→256*
            nn.Conv2d(64,32,3,padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2,mode='nearest'),  # →128×128
            nn.Upsample(scale_factor=2,mode='nearest'),  # →256×256 
            nn.Conv2d(32,3,3,padding=1)                   # final is 3×256×256
        )

    def forward(self, content, style):
        c_feat = self.enc(content)
        s_feat = self.enc(style)
        t = adain(c_feat, s_feat)
        out = self.dec(t)
        return out, t, s_feat

# ---------- Inference Wrapper ----------
class Stylizer(nn.Module):
    def __init__(self, model, style_dir, transform, device):
        super().__init__()
        self.model = model
        self.device = device
        self.style_feats = []
        style_dirs = sorted(os.listdir(style_dir))
        for label in style_dirs:
            cls_dir = os.path.join(style_dir, label)
            p = glob.glob(os.path.join(cls_dir, '*'))[0]
            img = transform(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad(): feat = self.model.enc(img)
            self.style_feats.append(feat)

    def forward(self, content, style_label):
        if isinstance(style_label, int):
            feat = self.style_feats[style_label].repeat(content.size(0),1,1,1)
        else:
            feat = torch.stack([self.style_feats[i] for i in style_label],0)
        t = adain(self.model.enc(content), feat)
        return self.model.dec(t)

# ---------- Training & Export ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--global_batch_size', type=int, default=32)
    parser.add_argument('--micro_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--precision', choices=['fp32','amp'], default='fp32')
    parser.add_argument('--strategy', choices=['none','ddp','fsdp'], default='none')
    parser.add_argument('--style_w', type=float, default=10.0)
    parser.add_argument('--tv_w', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--export_path', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    distributed = args.strategy in ['ddp','fsdp']
    if distributed:
        torch.distributed.init_process_group('nccl')
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        rank   = torch.distributed.get_rank()
        world  = torch.distributed.get_world_size()
        # rank = torch.distributed.get_rank(); world = torch.distributed.get_world_size()
    else:
        local_rank = rank = 0; world = 1; device = torch.device('cuda')
    assert args.global_batch_size % (args.micro_batch_size*world)==0
    accum = args.global_batch_size//(args.micro_batch_size*world)

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # Use env var if available, else fallback to --data_root
    data_root = os.getenv("IMG_DATA_DIR", args.data_root)

    ds = StyleTransferDataset(
        os.path.join(data_root, 'content'),
        os.path.join(data_root, 'style'),
        transform
    )
    sampler = DistributedSampler(ds) if distributed else None
    # loader = DataLoader(ds, batch_size=args.micro_batch_size, sampler=sampler,
    #                     shuffle=not distributed, num_workers=4, pin_memory=True)
    loader = DataLoader(
        ds,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        shuffle=not distributed,
        num_workers=16,                # more parallelism
        pin_memory=True,               # keep this on
        prefetch_factor=2,             # number of batches to prefetch per worker
    )

    device = torch.device('cuda')
    model = StyleTransferModel().to(device)
    opt = optim.Adam(model.dec.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if args.precision=='amp' else None

    # if args.strategy=='ddp': model=nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    if args.strategy == 'ddp':
        # torch.distributed.init_process_group('nccl')
        # torch.cuda.set_device(args.local_rank)                   # ← use local_rank
        # device = torch.device(f'cuda:{args.local_rank}')
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )
    if args.strategy=='fsdp': model=FSDP(model, cpu_offload=CPUOffload(False), sharding_strategy=ShardingStrategy.FULL_SHARD)

    ### Before we start training - start an MLFlow run
    try: 
        mlflow.end_run() # end pre-existing run, if there was one
    except:
        pass
    finally:
        mlflow.start_run(log_system_metrics=True) # Start MLFlow run
        # automatically log GPU and CPU metrics
        # Note: to automatically log AMD GPU metrics, you need to have installed pyrsmi
        # Note: to automatically log NVIDIA GPU metrics, you need to have installed pynvml
    
    # Let's get the output of rocm-info or nvidia-smi as a string...
    gpu_info = next(
        (subprocess.run(cmd, capture_output=True, text=True).stdout for cmd in ["nvidia-smi", "rocm-smi"] if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode == 0),
        "No GPU found."
    )
    # ... and send it to MLFlow as a text file
    mlflow.log_text(gpu_info, "gpu-info.txt")
    
    
    # Log hyperparameters - the things that we *set* in our experiment configuration
    # convert Namespace to dict
    hparams = vars(args)    # == {'lr': 0.0001, 'batch_size': 32, …}
    # load into MLflow
    mlflow.log_params(hparams)

    log=[]
    for e in range(args.epochs):
        if sampler: sampler.set_epoch(e)
        start=time.time(); opt.zero_grad()
        for i,(c,s,_) in enumerate(loader):
            c,s=c.to(device),s.to(device)
            with torch.cuda.amp.autocast(args.precision=='amp'):
                out,t,sf=model(c,s)
                of=model.enc(out)
                l_c=F.mse_loss(of,t)
                l_s=F.mse_loss(of.mean([2,3]), sf.mean([2,3]))+F.mse_loss(of.std([2,3]), sf.std([2,3]))
                l_tv = torch.sum(torch.abs(out[:,:,1:]-out[:,:,:-1]))+torch.sum(torch.abs(out[:,:,:,1:]-out[:,:,:,:-1]))
                loss=(l_c+args.style_w*l_s+args.tv_w*l_tv)/accum
            if scaler: scaler.scale(loss).backward()
            else: loss.backward()
            if (i+1)%accum==0:
                if scaler: scaler.step(opt); scaler.update()
                else: opt.step()
                opt.zero_grad()
        t_e=time.time()-start; m=torch.cuda.max_memory_allocated()/1e9
        if rank==0: log.append((args.strategy,world,e,t_e,m)); print(f"Epoch{e} {t_e:.1f}s mem{m:.2f}GB")
    
        # Log metrics - the things we *measure* - to MLFlow
        mlflow.log_metrics(
            {"epoch_time": t_e,
             "content_loss": l_c.item(),
             "style_loss": l_s.item(),
             "total_loss": loss.item(),
             }, step=e)
    
    if rank==0:
        base = model.module if hasattr(model,'module') else model
        stylizer = Stylizer(base, os.path.join(args.data_root,'style'), transform, device)
        scripted = torch.jit.script(stylizer)
        scripted.save(args.export_path)
        print(f"Exported stylizer to {args.export_path}")
        mlflow.log_artifact("./stylizer.pt", artifact_path="model")
        
    mlflow.end_run()

if __name__=='__main__': main()
