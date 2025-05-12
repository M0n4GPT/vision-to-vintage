import os
import time
import argparse
import subprocess
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models
from PIL import Image

import mlflow
import mlflow.pytorch

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import CPUOffload, ShardingStrategy
except ImportError:
    FSDP = None

# ---------- Configure MLFlow ----------
mlflow.set_tracking_uri("http://129.114.25.100:8080/")
mlflow.set_experiment("style-trans")

# ---------- Dataset ----------
class StyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_root, transform):
        self.content_paths = glob.glob(os.path.join(content_dir, '*'))
        self.style_paths = []  # list of (path, label)
        style_dirs = sorted(os.listdir(style_root))
        for idx, label in enumerate(style_dirs):
            cls_dir = os.path.join(style_root, label)
            if os.path.isdir(cls_dir):
                for p in glob.glob(os.path.join(cls_dir, '*')):
                    self.style_paths.append((p, idx))
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
            nn.Conv2d(64,32,3,padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(32,3,3,padding=1)
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
        for label in sorted(os.listdir(style_dir)):
            p = glob.glob(os.path.join(style_dir, label, '*'))[0]
            img = transform(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad(): feat = self.model.enc(img)
            self.style_feats.append(feat)

    def forward(self, content, style_label):
        if isinstance(style_label, int):
            feat = self.style_feats[style_label].repeat(content.size(0),1,1,1)
        else:
            feat = torch.stack([self.style_feats[i] for i in style_label], 0)
        t = adain(self.model.enc(content), feat)
        return self.model.dec(t)

# ---------- Training & Export ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--global_batch_size', type=int, default=128)
    parser.add_argument('--micro_batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--precision', choices=['fp32','amp'], default='fp32')
    parser.add_argument('--strategy', choices=['none','ddp','fsdp'], default='none')
    parser.add_argument('--style_w', type=float, default=10.0)
    parser.add_argument('--tv_w', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--export_path', type=str, required=True)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', type=int, default=0,
                        help='Local process rank for distributed training')
    args = parser.parse_args()

    # distributed setup
    distributed = args.strategy in ['ddp','fsdp']
    if distributed:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0; world = 1

    assert args.global_batch_size % (args.micro_batch_size * world) == 0
    accum = args.global_batch_size // (args.micro_batch_size * world)

    # transforms
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # data dirs
    data_root = os.getenv("IMG_DATA_DIR", args.data_root)
    train_content = os.path.join(data_root, 'random_inputs', 'random_train')
    val_content   = os.path.join(data_root, 'random_inputs', 'random_val')
    test_content  = os.path.join(data_root, 'random_inputs', 'random_test')
    train_styles  = os.path.join(data_root, 'train')
    val_styles    = os.path.join(data_root, 'val')
    test_styles   = os.path.join(data_root, 'test')

    # datasets
    train_ds = StyleTransferDataset(train_content, train_styles, transform)
    val_ds   = StyleTransferDataset(val_content,   val_styles,   transform)
    test_ds  = StyleTransferDataset(test_content,  test_styles,  transform)
    
    # samplers/loaders
    train_sampler = DistributedSampler(train_ds) if distributed else None
    train_loader  = DataLoader(train_ds, batch_size=args.micro_batch_size,
                               sampler=train_sampler, shuffle=(train_sampler is None),
                               num_workers=16, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.micro_batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    # model & optimizer
    model = StyleTransferModel().to(device)
    if args.fine_tune and args.pretrained_model:
        map_loc = {'cuda:%d' % 0: f'cuda:{args.local_rank}' } if distributed else None
        model.load_state_dict(torch.load(args.pretrained_model, map_location=map_loc))
    opt = optim.Adam(model.dec.parameters(), lr=args.lr)
    if args.strategy == 'ddp':
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    if args.strategy == 'fsdp':
        model = FSDP(model, cpu_offload=CPUOffload(False), sharding_strategy=ShardingStrategy.FULL_SHARD)

    scaler = torch.cuda.amp.GradScaler() if args.precision=='amp' else None

    # MLFlow run
    try: mlflow.end_run()
    except: pass
    mlflow.start_run(log_system_metrics=True)

    # log system and hyperparams
    gpu_info = next((subprocess.run(cmd, capture_output=True, text=True).stdout for cmd in [
                    "nvidia-smi","rocm-smi"] if subprocess.run(f"command -v {cmd}", shell=True).returncode==0),
                    "No GPU found.")
    mlflow.log_text(gpu_info, "gpu-info.txt")
    mlflow.log_params(vars(args))

    # training loop
    for e in range(args.epochs):
        if train_sampler: train_sampler.set_epoch(e)
        start = time.time()
        opt.zero_grad()
        for i, (c, s, _) in enumerate(train_loader):
            c, s = c.to(device), s.to(device)
            with torch.cuda.amp.autocast(args.precision=='amp'):
                out, t, sf = model(c, s)
                of = model.module.enc(out) if hasattr(model, 'module') else model.enc(out)
                l_c = F.mse_loss(of, t)
                l_s = F.mse_loss(of.mean([2,3]), sf.mean([2,3])) + F.mse_loss(of.std([2,3]), sf.std([2,3]))
                l_tv = torch.sum(torch.abs(out[:,:,1:]-out[:,:,:-1])) + torch.sum(torch.abs(out[:,:,:,1:]-out[:,:,:,:-1]))
                loss = (l_c + args.style_w*l_s + args.tv_w*l_tv) / accum
            if scaler: scaler.scale(loss).backward()
            else: loss.backward()
            if (i+1) % accum == 0:
                if scaler: scaler.step(opt); scaler.update()
                else: opt.step()
                opt.zero_grad()
        elapsed = time.time() - start
        mem=torch.cuda.max_memory_allocated()/1e9
        if rank == 0:
            print(f"Epoch {e} Time {elapsed:.1f}s mem{mem:.2f}GB")
            mlflow.log_metrics(
            {"epoch_time": elapsed,
             "train_content_loss": l_c.item(),
             "train_style_loss": l_s.item(),
             "train_total_loss": loss.item(),
             }, step=e)


    # export
    if rank == 0:
        # validation loop here
        model.eval()
        with torch.no_grad():
            val_losses = []
            for c, s, _ in val_loader:
                c, s = c.to(device), s.to(device)
                out, t, sf = model(c, s)
                of = model.module.enc(out) if hasattr(model, 'module') else model.enc(out)
                l_c = F.mse_loss(of, t)
                l_s = F.mse_loss(of.mean([2,3]), sf.mean([2,3])) + F.mse_loss(of.std([2,3]), sf.std([2,3]))
                l_tv = torch.sum(torch.abs(out[:,:,1:]-out[:,:,:-1])) + torch.sum(torch.abs(out[:,:,:,1:]-out[:,:,:,:-1]))
                val_losses.append((l_c + args.style_w*l_s + args.tv_w*l_tv).item())
            avg_val = sum(val_losses) / len(val_losses)
            print(f"Validation loss: {avg_val:.4f}")
            mlflow.log_metric('val_loss', avg_val, step=e)
        model.train()
                  
        base = model.module if hasattr(model, 'module') else model
        stylizer = Stylizer(base, train_styles, transform, device)
        scripted = torch.jit.script(stylizer)
        scripted.save(args.export_path)
        print(f"Exported stylizer to {args.export_path}")
        mlflow.log_artifact(args.export_path, artifact_path="model")
    mlflow.end_run()

if __name__ == '__main__':
    main()
