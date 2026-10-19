import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 128
D_conv = 5

# Field the wave propagates over (treated as a single-channel DxD image).
b = torch.randn(D, D, device=device) * 0.01

# Circular padding => the field lives on a torus, so wavefronts wrap around.
exc = nn.Conv2d(1, 1, kernel_size=D_conv, stride=1, padding=D_conv // 2,
                bias=False, padding_mode="circular").to(device)


def laplacian_kernel(size):
    """Isotropic 9-point discrete Laplacian embedded in a size x size kernel.

    Stencil (1/6) * [[1, 4, 1], [4, -20, 4], [1, 4, 1]] -- sums to zero (a
    second derivative). The 9-point form is near-isotropic, which keeps
    wavefronts circular instead of taking on the diamond anisotropy of the
    5-point stencil. Here it is the spatial coupling (diffusion) of an
    excitable medium, not a smoothing filter.
    """
    stencil = torch.tensor([[1., 4., 1.],
                            [4., -20., 4.],
                            [1., 4., 1.]], device=device) / 6.0
    k = torch.zeros(size, size, device=device)
    c = size // 2
    k[c - 1:c + 2, c - 1:c + 2] = stencil  # centered; zeros elsewhere
    return k


with torch.no_grad():
    exc.weight.copy_(laplacian_kernel(D_conv).view(1, 1, D_conv, D_conv))


def laplacian(field):
    """Apply exc (the discrete Laplacian) to a DxD field, torus boundaries."""
    x = field.unsqueeze(0).unsqueeze(0)  # (1, 1, D, D)
    return exc(x).squeeze(0).squeeze(0)


def excitable_step(u, v, a=0.75, b_thr=0.06, eps=0.02, dt=0.05,
                   Du=1.0, decay=0.0):
    """One explicit step of the Barkley excitable-medium model.

        u_t = Du * lap(u) + (1/eps) * u (1 - u) (u - (v + b_thr)/a) - decay * u
        v_t = u - v

    `u` is a fast activator, `v` a slow recovery variable. A superthreshold
    kick makes `u` jump toward 1 and diffuse into its neighbours -> a sharp
    traveling FRONT. Behind the front `v` rises, lifts the threshold, and
    forces `u` back to 0 -> a refractory tail that returns the medium to
    baseline. From a point stimulus: a single expanding ring, no oscillatory
    wake (unlike the linear wave equation).

    `decay` adds a linear leak to the activator. At 0 the front is regenerative
    and travels at constant amplitude; raise it and conduction becomes
    decremental -- the front dims as it travels and eventually extinguishes,
    i.e. a decaying traveling front.
    """
    u_th = (v + b_thr) / a
    react = u * (1.0 - u) * (u - u_th)
    u_next = (u + dt * (Du * laplacian(u) + react / eps - decay * u)).clamp(0.0, 1.0)
    v_next = v + dt * (u - v)
    return u_next, v_next


def sparse_binary(D, density=0.01, seed=0):
    """Random sparse binary DxD initial condition (on `device`)."""
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.rand(D, D, generator=g, device=device) < density).float()


def dilate(x, r=2):
    """Grow each set pixel into a (2r+1) square, torus wrap -> supra-threshold
    nuclei big enough to launch a wave instead of being pulled back by
    diffusion (the critical-nucleus effect)."""
    k = 2 * r + 1
    xp = F.pad(x[None, None], (r, r, r, r), mode="circular")
    return F.max_pool2d(xp, k, stride=1).squeeze()


def save_gif(path="wave.gif", steps=400, density=0.0006, seed=0, seed_r=2,
             dt=0.05, eps=0.02, a=0.75, b_thr=0.06, decay=0.0, amp_decay=0.0,
             fps=30):
    """Roll the excitable medium forward from sparse stimuli and save a GIF.

    `decay` is the physical activator leak (shortens how far a front travels
    before it extinguishes). `amp_decay` is a global temporal envelope,
    exp(-amp_decay * t), applied at readout: a regenerative excitable front has
    a fixed amplitude by construction, so a smoothly *decaying* traveling front
    can only come from an imposed envelope, not the autonomous dynamics.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Sparse point stimuli, dilated into small nuclei; medium starts at rest.
    u = sparse_binary(D, density=density, seed=seed)#, r=seed_r)
    v = torch.zeros(D, D, device=device)

    frames = []
    with torch.no_grad():
        for step in range(steps):
            env = math.exp(-amp_decay * step * dt)  # decaying amplitude in time
            frames.append((u * env).detach().to("cpu").numpy())
            u, v = excitable_step(u, v, a=a, b_thr=b_thr, eps=eps, dt=dt,
                                  decay=decay)
            u += sparse_binary(D, density=density)#, r=seed_r)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # u in [0,1]: bright = excited front, dark = rest/recovered baseline.
    im = ax.imshow(frames[0], cmap="inferno", vmin=0.0, vmax=1.0,
                   interpolation="nearest")

    def update(i):
        im.set_data(frames[i])
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {len(frames)}-frame GIF -> {path}")


if __name__ == "__main__":
    print(f"device: {device}")
    print("kernel (Laplacian):\n", exc.weight.view(D_conv, D_conv))

    # Diagnostic: sparse stimuli -> expanding excited fronts, recover behind.
    for decay in (0.0, 0.5):
        u = dilate(sparse_binary(D, density=0.0006, seed=0), r=2)
        v = torch.zeros(D, D, device=device)
        print(f"\n[decay={decay}]")
        for step in range(1, 201):
            u, v = excitable_step(u, v, decay=decay)
            if step % 40 == 0:
                excited = (u > 0.5).sum().item()
                print(f"  step {step:3d}: excited_cells={excited:5d} "
                      f"u_max={u.max():.3f} u_mean={u.mean():.4f} "
                      f"v_mean={v.mean():.4f}")

    # Sustained fronts (constant amplitude) and a decaying-amplitude variant.
    save_gif("wave.gif", steps=400, density=0.0006, seed=0, fps=30)
    save_gif("wave_decay.gif", steps=400, density=0.0006, seed=0, amp_decay=0.25, fps=60)
