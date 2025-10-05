import os
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from train_1 import StyleCLIP, load_generator

def load_mapper_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'mapper' not in ckpt:
        raise KeyError(f"Checkpoint at {ckpt_path} has no 'mapper' key.")
    model.mapper.load_state_dict(ckpt['mapper'])
    model.eval()
    return model

def make_side_by_side(orig_img, edited_img):
    w, h = orig_img.size
    combined = Image.new('RGB', (w * 2, h))
    combined.paste(orig_img, (0, 0))
    combined.paste(edited_img, (w, 0))
    return combined

def to_pil(tensor):
    img = ((tensor.clamp(-1, 1) + 1) * 0.5 * 255).round()
    arr = img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')
    return Image.fromarray(arr)

def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side images using a trained StyleCLIP mapper.")
    parser.add_argument('--ckpt_path', type=str, default='/StyleCLIP/Best_Checkpoints/Clinton_Best.pt')
    parser.add_argument('--w_plus', type=str, default='/StyleCLIP/celebrity_data/w_plus.npy')
    parser.add_argument('--stylegan_weights', type=str, default='/StyleCLIP/pretrained_models/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--ir_se50_weights', type=str, default='/StyleCLIP/pretrained_models/model_ir_se50.pth')
    parser.add_argument('--prompt', type=str, default='Hillary Clinton') #Use {celebrity_name} face for the prompt
    parser.add_argument('--alpha', type=float, default=0.065)
    parser.add_argument('--output_dir', type=str, default='inference_1_results')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(args.output_dir, exist_ok=True)

    class Opts: pass
    opts = Opts()
    opts.stylegan_weights = args.stylegan_weights
    opts.ir_se50_weights = args.ir_se50_weights
    opts.prompt = args.prompt

    model = StyleCLIP(opts, device).to(device)
    model.clip_model.float()
    model = load_mapper_checkpoint(model, args.ckpt_path, device)

    data_path = Path(args.w_plus)
    if data_path.suffix == '.npy':
        latents = np.load(data_path).astype(np.float32)
    else:
        lat = torch.load(data_path, map_location='cpu')
        latents = lat['latents'] if isinstance(lat, dict) and 'latents' in lat else lat
        latents = latents.cpu().numpy().astype(np.float32)

    print(f"Loaded {len(latents)} latents from {data_path}")

    for idx, w in enumerate(latents):
        w_tensor = torch.from_numpy(w).unsqueeze(0).to(device)
        with torch.no_grad():
            orig, _ = model.generator([w_tensor], input_is_latent=True, randomize_noise=False)
            w_edit, _, _ = model(w_tensor, args.prompt, args.alpha, device)
            edit, _ = model.generator([w_edit], input_is_latent=True, randomize_noise=False)

        combo = make_side_by_side(to_pil(orig), to_pil(edit))
        combo.save(Path(args.output_dir) / f"{idx:05d}.png")
        if idx % 50 == 0:
            print(f"Saved {idx:05d}.png")

    print("Done.")

if __name__ == '__main__':
    main()
