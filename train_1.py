import os
import argparse
from pathlib import Path
from itertools import cycle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
from PIL import Image
from torchvision import transforms
from models.facial_recognition.model_irse import Backbone
from models.facial_recognition.helpers import Flatten, l2_norm
from models.stylegan2.model import Generator

class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=512, lr_mul=0.01):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.lr_mul = lr_mul
        nn.init.normal_(self.fc1.weight, 0, 0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, 0, 0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x) * self.lr_mul, 0.2)
        return F.leaky_relu(self.fc2(x) * self.lr_mul, 0.2)

class SoftGatedMapper(nn.Module):
    def __init__(self, segments, latent_dim=512, hid_dim=512, depth=5):
        super().__init__()
        self.segments = segments
        self.mlps = nn.ModuleDict()
        for name, idxs in segments.items():
            mlp = [PixelNorm()]
            for _ in range(depth):
                mlp.append(EqualLinear(latent_dim, latent_dim, hid_dim=hid_dim, lr_mul=1)) # changes here from 1 to 0.1
            self.mlps[name] = nn.Sequential(*mlp)
        self.gate_logits = nn.Parameter(torch.zeros(2))
        self.gate_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, w_plus):
        
        gates = torch.sigmoid(self.gate_logits) * self.gate_scale
        pieces, seg_out = [], {}
        for name, idxs in self.segments.items():
            sub = w_plus[:, idxs]
            B, L, C = sub.shape
            delta = self.mlps[name](sub.view(-1, C)).view(B, L, C)
            if name == 'expr':
                delta = delta * gates[0]
                seg_out['expr'] = delta
            elif name == 'hair':
                delta = delta * gates[1]
                seg_out['hair'] = delta
            pieces.append(delta)
        delta_all = torch.cat(pieces, dim=1)
        return delta_all, gates, seg_out

class StyleCLIPMapper(nn.Module):
    def __init__(self, latent_dim=512, hid_dim=512):
        super().__init__()
        segments = {
            'coarse': list(range(0, 4)),
            'shape':  list(range(4, 8)),
            'expr':   list(range(8, 12)),
            'hair':   list(range(12, 16)),
            'micro':  list(range(16, 18)),
        }
        self.mapper = SoftGatedMapper(segments, latent_dim=latent_dim, hid_dim=hid_dim, depth=5)

    def forward(self, latents):
        return self.mapper(latents)

def load_generator(ckpt_path: Path, device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    g = Generator(1024, 512, 8).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    g.load_state_dict(state['g_ema'], strict=False)
    g.eval().requires_grad_(False)
    return g


def build_latent_loader(pt_path, batch_size, workers):
    data = torch.load(pt_path, map_location='cpu')
    if isinstance(data, torch.Tensor):
        latents = data.float()
    elif isinstance(data, np.ndarray):
        latents = torch.from_numpy(data).float()
    elif isinstance(data, dict) and 'latents' in data:
        latents = data['latents'].float()
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    ds = torch.utils.data.TensorDataset(latents)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


class IDLoss(nn.Module):
    def __init__(self, ir_se50_weights, device):
        super().__init__()
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(ir_se50_weights))
        self.pool = nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval().to(device)

    def extract(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]
        x = self.face_pool(x)
        return self.facenet(x)

    def forward(self, pred, orig):
        pred_feats = self.extract(pred)
        orig_feats = self.extract(orig).detach()
        sims = (pred_feats * orig_feats).sum(dim=-1)
        return (1 - sims).mean()

class StyleCLIP(nn.Module):
    def __init__(self, opts, device):
        super().__init__()
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device=device)
        self.generator = load_generator(Path(opts.stylegan_weights), device)
        self.id_loss = IDLoss(opts.ir_se50_weights, device)
        self.mapper = StyleCLIPMapper().to(device)

        for p in self.clip_model.parameters(): p.requires_grad = False
        for p in self.generator.parameters(): p.requires_grad = False

    def get_delta(self, w, prompt, device):
        tokens = clip.tokenize([prompt]).to(device)
        txt_feat = self.clip_model.encode_text(tokens)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            img, _ = self.generator([w], input_is_latent=True, randomize_noise=False)
            img = (img + 1) / 2
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img_feat = self.clip_model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        delta = txt_feat - img_feat
        delta_rep = delta.unsqueeze(1).repeat(1, w.size(1), 1)
        delta_w, gates, seg_out = self.mapper(delta_rep)
        return delta_w, gates, seg_out

    def forward(self, w, prompt, alpha, device):
        delta_w, gates, seg_out = self.get_delta(w, prompt, device)
        return w + alpha * delta_w, gates, seg_out

def train(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opts.output_dir, exist_ok=True)

    model = StyleCLIP(opts, device).train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.mapper.parameters()),
                           lr=opts.learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.iters_per_epoch)
    scaler = GradScaler()
    loader = build_latent_loader(opts.train_data, opts.batch, opts.workers)
    loader_iter = cycle(loader)
    global_step = 0

    for epoch in range(1, opts.epochs + 1):
        pbar = tqdm(range(opts.iters_per_epoch), desc=f"Epoch {epoch}/{opts.epochs}")
        for _ in pbar:
            w = next(loader_iter)[0].to(device)
            optimizer.zero_grad()
            with autocast(device.type):
                w_edited, gates, seg_out = model(w, opts.prompt, opts.alpha, device)
                delta_w = w_edited - w
                latent_l2 = delta_w.pow(2).mean()
                orig, _ = model.generator([w], input_is_latent=True, randomize_noise=False)
                edit, _ = model.generator([w_edited], input_is_latent=True, randomize_noise=False)
                orig_norm = (orig + 1) * 0.5
                edit_norm = (edit + 1) * 0.5
                id_l = model.id_loss(edit, orig)
                tokens = clip.tokenize([opts.prompt]).to(device)
                edit_img = F.interpolate(edit_norm, size=(224, 224), mode='bilinear', align_corners=False)
                img_feat = model.clip_model.encode_image(edit_img)
                txt_feat = model.clip_model.encode_text(tokens)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                clip_l = (1 - (img_feat * txt_feat).sum(dim=-1)).mean()
                loss_colour = opts.colour_lambda * F.l1_loss(edit_norm, orig_norm)
                expr_f = seg_out['expr'].view(w.size(0), -1)
                hair_f = seg_out['hair'].view(w.size(0), -1)
                loss_corr = opts.corr_lambda * (1 - F.cosine_similarity(expr_f, hair_f, dim=1).mean())
                dot = (expr_f * hair_f).sum(dim=1)
                loss_ortho = opts.ortho_lambda * dot.pow(2).mean()
                loss = (
                    opts.clip_lambda * clip_l +
                    opts.id_lambda * id_l +
                    opts.latent_lambda * latent_l2 +
                    loss_colour +
                    loss_corr +
                    loss_ortho
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.mapper.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            pbar.set_postfix(TL=f"{loss.item():.4f}", CL=f"{clip_l.item():.4f}", ID=f"{id_l.item():.4f}")
            if global_step % opts.image_interval == 0:
                with torch.no_grad():
                    o, _ = model.generator([w], input_is_latent=True, randomize_noise=False)
                    e, _ = model.generator([w_edited], input_is_latent=True, randomize_noise=False)
                o = ((o + 1) * 0.5).clamp(0, 1).cpu()
                e = ((e + 1) * 0.5).clamp(0, 1).cpu()
                for idx in range(e.shape[0]):
                    orig_img = transforms.ToPILImage()(o[idx])
                    edit_img = transforms.ToPILImage()(e[idx])
                    #orig_img.save(os.path.join(opts.output_dir, f"epoch{epoch}_step{global_step}_orig_{idx}.jpg"))
                    #edit_img.save(os.path.join(opts.output_dir, f"epoch{epoch}_step{global_step}_edit_{idx}.jpg"))
                    #Uncheck these to save separately and not concat
                    w_img, h_img = orig_img.size
                    concat_img = Image.new('RGB', (w_img * 2, h_img))
                    concat_img.paste(orig_img, (0, 0))
                    concat_img.paste(edit_img, (w_img, 0))
                    concat_img.save(os.path.join(opts.output_dir, f"epoch{epoch}_step{global_step}_concat_{idx}.jpg"))
            if global_step % opts.save_interval == 0:
                torch.save({
                    'step': global_step,
                    'mapper': model.mapper.state_dict(),
                    'opt': optimizer.state_dict(),
                    'sched': scheduler.state_dict()
                }, os.path.join(opts.output_dir, f"ckpt_{global_step}.pt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--train_data', type=str, default='/StyleCLIP/data/train_faces.pt')
    parser.add_argument('--stylegan_weights', type=str, default='/StyleCLIP/pretrained_models/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--ir_se50_weights', type=str, default='/StyleCLIP/pretrained_models/model_ir_se50.pth')
    parser.add_argument('--output_dir', type=str, default='train_1_results')
    parser.add_argument('--alpha', type=float, default=0.05) 
    parser.add_argument('--epochs', type=int, default=2) #
    parser.add_argument('--iters_per_epoch', type=int, default=300)
    parser.add_argument('--batch', type=int, default=8) #Change based on GPU (We trained on 3090)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--id_lambda', type=float, default=0.8)
    parser.add_argument('--clip_lambda', type=float, default=1)
    parser.add_argument('--latent_lambda', type=float, default=0.4) 
    parser.add_argument('--colour_lambda', type=float, default=0.01) 
    parser.add_argument('--corr_lambda', type=float, default=0.01) 
    parser.add_argument('--ortho_lambda', type=float, default=0.01) #Use these corr and ortho lambda to decorellate mappers
    parser.add_argument('--image_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=100)
    opts = parser.parse_args()
    train(opts)

if __name__ == '__main__':
    main()

"""

prompt: python train_1.py --prompt "Hillary Clinton" 

please change the path/parameters if needed from arg parser

"""
