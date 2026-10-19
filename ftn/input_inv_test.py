import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)
torch.set_default_device("cuda")
# ------------------------------------------------------------
# 1. Funky 2-D training distribution
# ------------------------------------------------------------

def sample_x(n):
    t = torch.rand(n) * 6 - 3
    branch = torch.where(torch.rand(n) < .5, -1., 1.)

    x0 = t + .12 * torch.randn(n)
    x1 = (
        .45 * (t**2 - 3)
        + branch * (.9 + .25 * torch.sin(3*t))
        + .12 * torch.randn(n)
    )
    return torch.stack([x0, x1], dim=1)


def true_f(x):
    return (
        torch.sin(1.7 * x[:, 0])
        + .35 * x[:, 1]
        + .15 * x[:, 0] * x[:, 1]
    ).unsqueeze(1)


# ------------------------------------------------------------
# 2. 8-wide, 8-layer teacher
# ------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 8)
        self.hidden = nn.ModuleList([nn.Linear(8, 8) for _ in range(7)])
        self.out = nn.Linear(8, 1)

    def forward(self, x, return_acts=False):
        acts = []

        x = torch.tanh(self.first(x))
        acts.append(x)

        for layer in self.hidden:
            x = torch.tanh(layer(x))
            acts.append(x)

        y = self.out(x)

        if return_acts:
            return y, acts
        return y


teacher = MLP()

x_train = sample_x(4096)
y_train = true_f(x_train)

opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)

for step in range(3000):
    i = torch.randint(len(x_train), (256,))
    loss = F.mse_loss(teacher(x_train[i]), y_train[i])

    opt.zero_grad()
    loss.backward()
    opt.step()

print("teacher MSE:", loss.item())


# ------------------------------------------------------------
# 3. Store tiny activation-distribution sketch
#
# Mean + covariance + random Fourier feature means.
# No input examples need to survive after this.
# ------------------------------------------------------------

RFF = 32

# These can be globally fixed for every teacher.
torch.manual_seed(123)
rff_params = []

for layer in range(8):
    W = 2.0 * torch.randn(RFF, 8)
    b = 2 * math.pi * torch.rand(RFF)
    rff_params.append((W, b))


with torch.no_grad():
    _, acts = teacher(x_train, return_acts=True)

stats = []

for z, (W, b) in zip(acts, rff_params):
    mu = z.mean(0)

    centered = z - mu
    cov = centered.T @ centered / (len(z) - 1)

    # Kernel-distribution fingerprint
    phi = torch.cos(z @ W.T + b)
    rff_mean = phi.mean(0)

    stats.append((mu, cov, rff_mean))


# Training data is conceptually gone.
del x_train, y_train, acts


# ------------------------------------------------------------
# 4. Start from random noise and optimize INPUTS
# ------------------------------------------------------------

x_fake = 2 * torch.randn(1024, 2)
x_fake.requires_grad_()

opt_x = torch.optim.Adam([x_fake], lr=0.05)

for step in range(2000):

    _, acts = teacher(x_fake, return_acts=True)

    inversion_loss = 0.

    for z, (mu, cov, rff_mean), (W, b) in zip(
        acts, stats, rff_params
    ):
        fake_mu = z.mean(0)

        centered = z - fake_mu
        fake_cov = centered.T @ centered / (len(z) - 1)

        fake_rff = torch.cos(z @ W.T + b).mean(0)

        inversion_loss += (fake_mu - mu).pow(2).mean()
        inversion_loss += .2 * (fake_cov - cov).pow(2).mean()
        inversion_loss += 2.0 * (fake_rff - rff_mean).pow(2).mean()

    # Weak regularizer just prevents extreme input values.
    inversion_loss += 1e-4 * x_fake.pow(2).mean()

    opt_x.zero_grad()
    inversion_loss.backward()
    opt_x.step()


x_fake = x_fake.detach()


# ------------------------------------------------------------
# 5. Evaluation ONLY.
#
# In a real data-free setting you would not have this.
# Here we resample the known distribution so we can measure
# how well reconstruction worked.
# ------------------------------------------------------------

x_real = sample_x(2000)


def mmd(x, y, sigma=1.0):
    """RBF maximum mean discrepancy."""
    xx = torch.exp(-torch.cdist(x, x).pow(2) / (2 * sigma**2)).mean()
    yy = torch.exp(-torch.cdist(y, y).pow(2) / (2 * sigma**2)).mean()
    xy = torch.exp(-torch.cdist(x, y).pow(2) / (2 * sigma**2)).mean()
    return xx + yy - 2 * xy


def hist2d(x, bins, lo, hi):
    scale = bins / (hi - lo)

    i = ((x[:, 0] - lo) * scale).long().clamp(0, bins - 1)
    j = ((x[:, 1] - lo) * scale).long().clamp(0, bins - 1)

    h = torch.bincount(i * bins + j, minlength=bins*bins).float()
    h += 1e-6
    return h / h.sum()


# MMD on subsets to keep cdist small.
p = x_real[:1000]
q = x_fake[:1000]

print("MMD^2:", mmd(p, q).item())

# Crude joint histogram KL.
P = hist2d(x_real, 40, -4., 4.)
Q = hist2d(x_fake, 40, -4., 4.)

kl_pq = (P * (P / Q).log()).sum()
kl_qp = (Q * (Q / P).log()).sum()

print("symmetric histogram KL:", (0.5 * (kl_pq + kl_qp)).item())


# ------------------------------------------------------------
# 6. Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(x_real[:, 0].cpu(), x_real[:, 1].cpu(), s=3)
plt.title("Real input distribution")
plt.xlim(-4, 4)
plt.ylim(-3, 4)

plt.subplot(1, 2, 2)
plt.scatter(x_fake[:, 0].cpu(), x_fake[:, 1].cpu(), s=3)
plt.title("Recovered from noise")
plt.xlim(-4, 4)
plt.ylim(-3, 4)

plt.tight_layout()
plt.savefig("input_inv_test.png")