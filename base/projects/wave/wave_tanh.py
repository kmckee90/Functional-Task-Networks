"""Local tanh RNN on a 2D sheet, with an exact spectral radius.

    u_next = tanh(rho * ((2 - omega**2)*u + c**2*L(u) - v) + input)
    v_next = tanh(rho * u)

Two channels per pixel provide the memory needed for traveling ripples.
L is the 9-point Laplacian, whose eigenvalues ell lie in [-16/3, 0]
for zeros, circular, and reflect padding. These are separable boundaries:
L = (Tx*Ty + 4*Tx + 4*Ty - 20*I)/6, where each 1D neighbor-sum
operator T has real eigenvalues in [-2, 2].
Each mode of the full recurrence W is rho * [[a, -1], [1, 0]], where
 a = 2 - omega**2 + c**2*ell.
For 0 < omega**2 and omega**2 + (16/3)*c**2 < 4, we have -2 < a < 2.
The roots of z**2 - a*z + 1 are conjugates of modulus one. Consequently
EVERY eigenvalue of W has modulus rho for all three padding choices,
provided the parameter conditions above hold.

rho < 1 gives asymptotic decay near zero, where tanh is linear. This is
not a global contraction claim: large inputs saturate, and wave collisions
are nonlinear. Spectral radius refers to W (also the Jacobian at zero).

Run: python wave_tanh.py --rho 0.985 --output wave.gif
Dependencies: torch, numpy, Pillow. Importing does not run the demo.
"""

import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F


class WaveTanh(nn.Module):
    """Fixed 3x3 connectivity; state shape (batch, 2, height, width).

    Use model(state, perturbation) to inject an input of shape
    (batch, 1, height, width), or model(state) for autonomous evolution.
    The kernel is a buffer, with no trainable parameters.
    """

    def __init__(self, rho=0.985, c=0.65, omega=0.15, padding="circular"):
        super().__init__()
        if padding not in ("zeros", "circular", "reflect"):
            raise ValueError("padding must be zeros, circular, or reflect")
        self.padding = padding
        # if not math.isfinite(rho) or not 0 < rho < 1:
        #     raise ValueError("rho must be in (0, 1) for decaying linear modes")
        # if not (math.isfinite(c) and math.isfinite(omega)
        #         and c > 0 and omega > 0
        #         and omega**2 + (16 / 3)*c**2 < 4):
        #     raise ValueError("require c, omega > 0 and omega² + (16/3)c² < 4")
        self.rho, self.c, self.omega = rho, c, omega
        lap = torch.tensor([[1., 4., 1.], [4., -20., 4.], [1., 4., 1.]]) / 6
        kernel = torch.zeros(2, 2, 3, 3)
        kernel[0, 0] = c**2 * lap
        kernel[0, 0, 1, 1] += 2 - omega**2
        kernel[0, 1, 1, 1] = -1
        kernel[1, 0, 1, 1] = 1
        self.register_buffer("kernel", rho * kernel)

    def forward(self, state, perturbation=None):
        """One step of h_next = tanh(W h + B input).

        zeros: outside values are zero; circular: wrap opposite edges;
        reflect: mirror interior values (requires height and width >= 2).
        """
        if state.ndim != 4 or state.shape[1] != 2:
            raise ValueError("state must have shape (batch, 2, height, width)")
        if self.padding == "reflect" and min(state.shape[-2:]) < 2:
            raise ValueError("reflect padding requires height and width >= 2")
        mode = "constant" if self.padding == "zeros" else self.padding
        pre = F.conv2d(F.pad(state, (1, 1, 1, 1), mode=mode), self.kernel)
        if perturbation is not None:
            if perturbation.shape != state[:, :1].shape:
                raise ValueError("perturbation must match state[:, :1].shape")
            pre = pre + torch.cat((perturbation, torch.zeros_like(perturbation)), dim=1)
        return torch.tanh(pre)


@torch.no_grad()
def save_gif(path="wave.gif", rho=0.985, size=128, steps=180, fps=25,
             device="cpu", padding="circular"):
    """Three initial 1-bits, no subsequent forcing or readout decay.

    Fixed signed logarithmic colors reveal faint ripples without rescaling
    each frame: red negative, black zero, blue positive.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    if size < 8 or steps < 2 or not 1 <= fps <= 100:
        raise ValueError("require size >= 8, steps >= 2, and 1 <= fps <= 100")
    model = WaveTanh(rho=rho, padding=padding).to(device)
    state = torch.zeros(1, 2, size, size, device=device)
    for y, x in ((size // 3, size // 3),
                #  (size // 2, 3*size // 4),
                #  (3*size // 4, 2*size // 5),
                 ):
        state[0, 0, y, x] = 0.0

    palette = []
    for i in range(256):
        s = (i - 128) / (128 if i < 128 else 127)
        intensity = round(255 * abs(s))
        palette.extend((intensity, 0, 0) if s < 0 else (0, 0, intensity))
    frames = []
    peaks = []
    state0 = state.clone()

    mask = torch.zeros_like(state0)
    mask_idx = torch.topk(torch.rand(size*size), k=64).indices
    mask.view(1,2,-1)[:, 0, mask_idx] = 1

    for step in range(steps):
        if step % 4 == 0:
            # breakpoint()
            state += torch.bernoulli(state0+0.5)* torch.randn_like(state0)  * mask * 0.1
            # state += mask*0.25

        field = state[0, 0].cpu().numpy()
        peak = float(np.abs(field).max())
        peaks.append(peak)
        color = np.sign(field) * np.log1p(np.abs(field) / 0.002) / np.log1p(500)
        indices = np.clip(np.rint(128 + np.where(color < 0, 128, 127)*color),
                          0, 255).astype(np.uint8)
        sheet = Image.fromarray(indices).convert("P")
        sheet.putpalette(palette)
        sheet = sheet.resize((512, 512), Image.Resampling.NEAREST)
        frame = Image.new("P", (512, 556), 128)
        frame.putpalette(palette)
        frame.paste(sheet, (0, 44))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 5), f"t={step:03d}   rho={rho:g}   max |u|={peak:.5f}   {padding}", fill=0)
        draw.text((8, 23), "red: negative   black: zero   blue: positive | fixed log scale", fill=0)
        frames.append(frame)
        state = model(state)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=round(100 / fps)*10, loop=0, optimize=False, disposal=2)
    print(f"Saved {path}: {steps} frames; spectral radius = {rho:g}; "
          f"max |u| {peaks[0]:.5f} -> {peaks[-1]:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, default=0.999)
    parser.add_argument("--output", default="wave.gif")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--padding", choices=("zeros", "circular", "reflect"),
                        default="circular", help="boundary padding (default: circular)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    save_gif(args.output, args.rho, args.size, args.steps, args.fps, args.device,
             args.padding)
