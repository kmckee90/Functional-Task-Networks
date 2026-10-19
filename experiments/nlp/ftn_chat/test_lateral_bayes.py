"""Run on the pod: lateral parity, bypass, posterior and archive invariants."""
from dataclasses import replace
import torch
from config import CFG
from spatial import kwta, lateral_kwta, LateralSmoother
import test_bayes_configurer as posterior_tests
import route_repair
from input_routing import InputRoutingArchive
from lora_bank import MaskController
from bayes_configurer import BayesianPerplexityConfigurer


def main():
    cfg = replace(CFG, grid_dim=8, k=8, lateral_steps=4,
                  lateral_kernel=3, lateral_polyak=0.0,
                  lateral_noise_sd=0.0, torus=True)
    for device in ("cpu", "cuda"):
        torch.manual_seed(12)
        x = torch.randn(64, device=device)
        for disabled in (replace(cfg, lateral_steps=0), replace(cfg, lateral_polyak=1.0)):
            disabled.lateral_noise_sd = 10.0
            before = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
            assert torch.equal(lateral_kwta(x, 8, disabled), kwta(x, 8))
            after = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
            assert torch.equal(before, after)
        for torus in (False, True):
            for polyak in (0.0, 0.4):
                c = replace(cfg, torus=torus, lateral_polyak=polyak)
                smoother = LateralSmoother(8, 3, torus)
                smoother.conv.to(device)
                expected = kwta(x, 8)
                for _ in range(4):
                    expected = polyak * expected + (1-polyak) * smoother.step(expected)
                actual = lateral_kwta(x, 8, c)
                assert torch.equal(actual, kwta(expected, 8))
                assert actual.sum() == 8
        assert not torch.equal(lateral_kwta(x, 8, cfg), kwta(x, 8))

    # User's production spatial dimensions and kernel, including lateral noise.
    production = replace(CFG, lateral_steps=4, lateral_kernel=17,
                         lateral_polyak=0.0, lateral_noise_sd=0.1, torus=True)
    x = torch.randn(production.H, device="cuda")
    result = lateral_kwta(x, production.k, production)
    assert result.sum() == production.k and torch.isfinite(result).all()

    posterior_tests.test_posterior()
    posterior_tests.test_routed_refinement()
    posterior_tests.CFG = replace(posterior_tests.CFG, lateral_steps=2, lateral_kernel=3)
    posterior_tests.test_posterior()

    # Exercise repair and verify direct recall is the exact validated mask,
    # including the no-improvement baseline stored for a brand-new route.
    ctrl = MaskController(64, 8)
    model = posterior_tests.TinyModel(ctrl)
    cfgr = BayesianPerplexityConfigurer(cfg, ctrl)
    cfg.input_route_steps = 2
    cfg.input_route_restarts = 1
    archive = InputRoutingArchive(cfg, None)
    examples = [{"ids": torch.tensor([1, 2, 3])}]
    route_repair._examples = lambda *a: (examples, examples, ["uniqueftn"], ["uniqueftn"] * 3)
    def answer_ce(model, ctrl, ex, mask):
        ctrl.set_mask(mask)
        return model(ex["ids"]).loss
    route_repair._answer_ce = answer_ce
    stats = route_repair.repair_routes(model, ctrl, cfgr, archive, ["task"], None, cfg)
    mask, info = archive.match("uniqueftn")
    assert info["reason"] == "matched" and mask.sum() == 8
    assert abs(float(answer_ce(model, ctrl, examples[0], mask).detach()) - stats["task"]["after_ce"]) < 1e-6
    print("PASS: lateral parity, exact bypass/RNG, posterior gradients/rollback, repaired route recall")


if __name__ == "__main__":
    main()
