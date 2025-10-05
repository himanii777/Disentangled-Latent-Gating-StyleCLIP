import os
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from train_2 import StyleCLIP, load_generator   


def load_mapper_checkpoint(model, ckpt_path: str | Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "mapper" not in ckpt:
        raise KeyError(f"{ckpt_path} is missing a 'mapper' key.")
    model.mapper.load_state_dict(ckpt["mapper"])
    model.eval().requires_grad_(False)
    return model


def make_side_by_side(orig: Image.Image, edit: Image.Image) -> Image.Image:
    w, h = orig.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(orig, (0, 0))
    canvas.paste(edit, (w, 0))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', type=str, default='/StyleCLIP/Best_Checkpoints/Suprised.pt')
    p.add_argument('--w_plus', type=str, default='/StyleCLIP/celebrity_data/w_plus.npy')
    p.add_argument("--stylegan_weights",
                   type=str,
                   default="/StyleCLIP/pretrained_models/stylegan2-ffhq-config-f.pt")
    p.add_argument("--ir_se50_weights",
                   type=str,
                   default="/StyleCLIP/pretrained_models/model_ir_se50.pth")
    p.add_argument('--prompt', type=str, default='Surprised Face')
    p.add_argument("--alpha",     type=float, default=0.03)  
    p.add_argument("--output_dir",type=str,   default="Inference_2_results")
    p.add_argument("--device",    type=str,   choices=["cuda", "cpu"], default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    class _Opts:
        stylegan_weights: str
        ir_se50_weights:  str
        prompt:           str
    opts = _Opts()
    opts.stylegan_weights = args.stylegan_weights
    opts.ir_se50_weights  = args.ir_se50_weights
    opts.prompt           = args.prompt
    model = StyleCLIP(opts, device).to(device)
    model.clip_model.float()                    
    model = load_mapper_checkpoint(model, args.ckpt_path, device)
    lat_path = Path(args.w_plus)
    if lat_path.suffix == ".npy":
        latents = np.load(lat_path).astype(np.float32)
    else:
        buf = torch.load(lat_path, map_location="cpu")
        latents = (buf["latents"] if isinstance(buf, dict) and "latents" in buf else buf
                   ).cpu().numpy().astype(np.float32)
    print(f"Loaded {len(latents)} latent codes from {lat_path}")

    def _to_pil(t: torch.Tensor) -> Image.Image:
        img = ((t.clamp(-1, 1) + 1) * 127.5).round()
        arr = img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype("uint8")
        return Image.fromarray(arr)

    for idx, w in enumerate(latents):
        w = torch.from_numpy(w).unsqueeze(0).to(device)
        with torch.no_grad():
            orig, _  = model.generator([w], input_is_latent=True, randomize_noise=False)
            w_edit, _, _ = model(w, args.prompt, args.alpha, device)
            edit, _  = model.generator([w_edit], input_is_latent=True, randomize_noise=False)

        combo = make_side_by_side(_to_pil(orig), _to_pil(edit))
        out_fp = Path(args.output_dir) / f"{idx:05d}.png"
        combo.save(out_fp)
        if idx % 50 == 0:
            print(f"Saved {out_fp}")

    print("Done.")


if __name__ == "__main__":
    main()
