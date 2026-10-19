"""Animate actual FTN processing: white mask pixels over red/blue wave fields.

Run in the base pod:
    python animate_wave.py --config WaveDemo --samples 3
    python animate_wave.py --config WaveDemo WaveSharedMask WaveDetach
    python animate_wave.py --checkpoint artifacts/my_run/model.pt

Without a checkpoint, initializes an FTN, optionally trains on one task, then
searches its mask on a support batch. Animated inputs are a fresh held-out batch.
Every sample begins with a blank reservoir; all processing/wave microsteps come
from WaveFTN.trace, not a separate simulation. White is the current mask, blue
positive r[:,0], red negative, black zero. Colors use one fixed signed-log scale
across every frame/sample. A PNG strip and JSON provenance accompany the GIF.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

import config
from run_compose import build_model, config_from_dict, effective_config_dict
from tasks import DataGenSine, make_rng
from wave_ftn import WaveFTN


def make_data(cfg):
    if cfg.TASK == "compose":
        from task_comp import DataGenComposition
        return DataGenComposition(n_tasks=cfg.N_TASK, n_samples=cfg.T,
                                  rng=make_rng(cfg.SEED+100), seed=cfg.SEED, **cfg.COMP_PARS)
    return DataGenSine(input_dim=cfg.X_DIM, output_dim=cfg.Y_DIM,
                       n_tasks=cfg.N_TASK, task_size=cfg.TASK_SIZE,
                       n_samples=cfg.T, frequency=cfg.FREQUENCY,
                       classify=cfg.CLASSIFICATION, data_temp=cfg.DATA_TEMP,
                       correlated_tasks=cfg.CORRELATED_TASKS, concept_shift=cfg.CONCEPT_SHIFT,
                       domain_shift_factor=cfg.DOMAIN_SHIFT_FACTOR,
                       domain_shift=cfg.DOMAIN_SHIFT, domain_shift_stride=cfg.DOMAIN_SHIFT_STRIDE,
                       domain_dense_projection=cfg.DOMAIN_DENSE_PROJECTION,
                       domain_per_task_projection=cfg.DOMAIN_PER_TASK_PROJECTION,
                       rng=make_rng(cfg.SEED+100))


def save_animation(model, x, path, fps=12, title="Wave FTN"):
    output, trace = model.trace(x)
    peak = max(frame['r'][:,0].abs().max().item() for frame in trace)
    peak = max(peak, 1e-8)
    floor = peak/100
    side = 512
    frames, durations, strip = [], [], []
    for sample in range(len(x)):
        for snapshot in trace:
            field = snapshot['r'][sample,0].cpu().numpy()
            intensity = np.log1p(np.abs(field)/floor)/np.log1p(peak/floor)
            rgb = np.zeros((*field.shape, 3), dtype=np.uint8)
            rgb[...,0] = np.rint(255*intensity*(field<0)).astype(np.uint8)
            rgb[...,2] = np.rint(255*intensity*(field>0)).astype(np.uint8)
            mask = snapshot['mask'].reshape(model.square_dim,model.square_dim).cpu().numpy()>0.5
            rgb[mask] = 255
            sheet = Image.fromarray(rgb).resize((side,side), Image.Resampling.NEAREST)
            image = Image.new('RGB',(side,side+64),'black')
            image.paste(sheet,(0,64))
            draw = ImageDraw.Draw(image)
            draw.text((8,5),title,fill='white')
            detail = ('reset: blank reservoir' if snapshot['phase']=='reset' else
                      f"step {snapshot['step']+1}/{model.n_stack}, "
                      f"wave substep {snapshot['substep']}/{model.wave_substeps}")
            draw.text((8,23),f'sample {sample+1}/{len(x)} | {detail}',fill='white')
            draw.text((8,42),'white: mask | red: negative | blue: positive | fixed log scale',fill='white')
            frames.append(image)
            durations.append(500 if snapshot['phase']=='reset' else round(100/fps)*10)
            if sample==0 and snapshot['substep']==model.wave_substeps:
                strip.append(image.resize((256,288),Image.Resampling.NEAREST))
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=durations,loop=0,disposal=2)
    montage = Image.new('RGB',(256*len(strip),288),'black')
    for i,panel in enumerate(strip):
        montage.paste(panel,(256*i,0))
    montage.save(path.with_suffix('.png'))
    return dict(samples=len(x), frames=len(frames), processing_steps=model.n_stack,
                wave_substeps=model.wave_substeps, wave_peak=peak,
                masks=model.m.cpu().tolist(), predictions=output.cpu().tolist(),
                inputs=x.cpu().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', nargs='+', default=['WaveDemo'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/wave_demo.gif'))
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--train-batches', type=int, default=0)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.samples<1 or args.train_batches<0 or not 1<=args.fps<=100:
        parser.error('require samples>=1, train-batches>=0, and 1<=fps<=100')
    torch.set_num_threads(4)
    torch.set_default_device(args.device)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint,map_location=args.device,weights_only=False)
        cfg = config_from_dict(checkpoint['config_dict'])
    else:
        cfg = type('AnimationConfig',tuple(getattr(config,c) for c in reversed(args.config)),{})()
    if cfg.ARCHITECTURE != 'wave':
        parser.error('animation requires ARCHITECTURE="wave"')
    if not 0<=args.task<cfg.N_TASK or args.samples>cfg.T:
        parser.error('task must be in range and samples must not exceed config T')
    torch.manual_seed(cfg.SEED)
    data = make_data(cfg)
    model = build_model(cfg,data.total_input_dim,data.output_dim)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg.TRAIN_LR)
    for _ in range(args.train_batches):
        x,y = data(args.task)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        model.loss(model(x,y),y).backward()
        optimizer.step()
    model.eval()
    model._reconfig(*data(args.task))
    x,_ = data(args.task)
    label = ('checkpoint' if checkpoint else f'demo: {args.train_batches} training batches')
    label += ' | '+('per-step masks' if model.step_masks else 'shared mask')
    info = save_animation(model,x[:args.samples],args.output,args.fps,label)
    info.update(config=effective_config_dict(cfg),checkpoint=str(args.checkpoint),
                train_batches=args.train_batches,task=args.task)
    args.output.with_suffix('.json').write_text(json.dumps(info,indent=2))
    print(f'Saved {args.output}: {info["samples"]} samples, {info["frames"]} frames')


if __name__=='__main__':
    main()
