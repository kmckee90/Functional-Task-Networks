"""Remote correctness checks; these synthetic probes are never report results."""
import argparse
import importlib
from types import SimpleNamespace

import torch
from torch import nn

from ftn_models import FusedBankLinear, VARIANTS, build, load_family


def check_contraction(cfg):
    cfg = SimpleNamespace(LORA_RANK=2, LORA_ALPHA=4., LORA_B0_SCALE=.01,
                          LORA_BIAS=True, LORA_DROPOUT=0.)
    controller = SimpleNamespace(H=7, k=3, detach_atoms=False,
                                 mask=torch.tensor([1., 0., 1., 0., 0., 1., 0.], requires_grad=True))
    base = nn.Linear(5, 4).double()
    layer = FusedBankLinear(base, controller, cfg).double()
    controller.mask = controller.mask.double().detach().requires_grad_(True)
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    actual = layer(x)
    flat = x.reshape(-1, 5)
    proj = torch.einsum('bi,hir->bhr', flat, layer.A)
    delta = torch.einsum('bhr,hro->bo', proj * controller.mask[None, :, None], layer.B)
    expected = layer.base(flat) + (cfg.LORA_ALPHA/controller.k) * (delta + controller.mask @ layer.bias_bank)
    expected = expected.reshape(2, 3, 4)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    pars = [x, controller.mask, layer.A, layer.B, layer.bias_bank]
    ga = torch.autograd.grad(actual.square().sum(), pars, retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), pars)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-9)
    assert ga[1][controller.mask == 0].abs().sum().item() > 0
    controller.detach_atoms = True
    loss = layer(x).sum()
    atom_grads = torch.autograd.grad(loss, [layer.A, layer.B], allow_unused=True)
    assert atom_grads == (None, None)
    print('PASS: token-bank contraction, every mask gradient, detached search atoms')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--full', action='store_true')
    args = p.parse_args()
    torch.set_default_device('cuda')
    torch.manual_seed(917)
    torch.set_num_threads(4)
    cfg, runner = load_family(args.variant)
    check_contraction(cfg)
    if not args.full:
        cfg.SQUARE_DIM, cfg.K = 8, 8
        cfg.LAYERS, cfg.INNER_DIM = 2, 4
        cfg.LORA_WIDTH, cfg.LORA_LAYERS = 16, 2
        cfg.RECONFIG_STEPS = 3
        cfg.BAYES_CHANGEPOINT_RECONFIG_STEPS = 3
        cfg.EVAL_RECONFIG_STEPS = 3
    cfg.N_TASK, cfg.T, cfg.TRAIN_DURATION = 2, 16, 12
    cfg.RECONFIG_SAMPLE_SIZE = 16
    cfg.N_RETRO_EVAL_BATCHES = 1
    cfg.RENDER_STEPS = False
    cfg.EVAL_PROSPECTIVE = True
    cfg.N_SAMPLE_PER_TASK = min(cfg.N_SAMPLE_PER_TASK, 2)
    cfg.REPLAY_BATCH_SIZE = min(cfg.REPLAY_BATCH_SIZE, 16)
    runner.Config = cfg
    model = build(cfg, 4, 3)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.TRAIN_LR)
    replay = runner.ReplayBuffer(cfg.REPLAY_MAXLEN, batch_size=cfg.REPLAY_BATCH_SIZE)
    # Fixed teacher; separate fresh draws exercise both native train and eval.
    weights = torch.randn(2, 4, 3)
    def data(task):
        x = torch.randn(cfg.T, 4)
        y = torch.nn.functional.one_hot((x @ weights[task]).argmax(-1), 3).float()
        return x, y
    collector = runner.metrics_mod.MetricsCollector()
    runner.train(model, data, [0, 1], 1, 'correctness', opt, replay,
                 collector=collector, save_viz=False)
    assert len(collector.rows) == 2
    assert all(torch.isfinite(v).all() for v in model.parameters())
    assert int(model.m.sum()) == cfg.K
    if hasattr(model, 'get_reconfig_state'):
        before = model.get_reconfig_state()
        runner.evaluate_backwards(model, data, [0, 1])
        after = model.get_reconfig_state()
        for k in before:
            torch.testing.assert_close(before[k], after[k], rtol=0, atol=0)
    print('PASS:', args.variant, 'full' if args.full else 'small', collector.summary())


if __name__ == '__main__':
    main()
