"""Read-only checkpoint probe for teacher-forced fit versus free recall."""
import json
import os

import torch
import torch.nn.functional as F

torch.set_default_device("cuda")

from bayes_configurer import BayesianPerplexityConfigurer, ReactiveState
from config import CFG
from data import format_distill
from model import build, load_bank
from train import _gather_answer_logits


def score(model, ctrl, ex, teacher, mask):
    ctrl.set_mask(mask)
    ids = ex["ids"].unsqueeze(0).to("cuda")
    labels = ex["lbl"].unsqueeze(0).to("cuda")
    with torch.no_grad():
        student, targets = _gather_answer_logits(model(input_ids=ids).logits, labels)
        tlogp = F.log_softmax(teacher.float(), -1)
        slogp = F.log_softmax(student.float(), -1)
        kl = (tlogp.exp() * (tlogp - slogp)).sum(-1)
        ce = F.cross_entropy(student.float(), targets, reduction="none")
        hit = (student.argmax(-1) == targets).float()
    return dict(kl=kl.mean().item(), kl8=kl[:8].mean().item(),
                ce=ce.mean().item(), ce8=ce[:8].mean().item(),
                hit=hit.mean().item(), hit8=hit[:8].mean().item())


def generate(model, tok, ctrl, mask, question, thinking, limit=64):
    ctrl.set_mask(mask)
    enc = tok.apply_chat_template(
        [{"role": "system", "content": CFG.qa_sys},
         {"role": "user", "content": question}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=thinking,
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=limit, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)


def main():
    print("CKPT", CFG.ckpt_dir, flush=True)
    model, tok, ctrl, wrappers, _ = build(CFG)
    load_bank(wrappers, CFG.ckpt_dir / "bank.pt")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    state = json.loads((CFG.ckpt_dir / "state.json").read_text())
    cfgr = BayesianPerplexityConfigurer(CFG, ctrl, state.get("reactive"))
    cfgr.load_state_dict(torch.load(CFG.ckpt_dir / "posterior.pt", map_location="cpu", weights_only=True))
    original = cfgr.state_dict()
    detector = cfgr.detector.to_dict()
    print("TRAINED", state["trained"], "DETECTOR", detector, flush=True)
    print("TEMPLATE_TRUE", repr(tok.apply_chat_template(
        [{"role":"system","content":CFG.qa_sys},{"role":"user","content":"Test?"}],
        add_generation_prompt=True, tokenize=False)[-110:]), flush=True)
    print("TEMPLATE_FALSE", repr(tok.apply_chat_template(
        [{"role":"system","content":CFG.qa_sys},{"role":"user","content":"Test?"}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)[-110:]), flush=True)

    root = CFG.data_cache / "qa"
    for key in ("base_ftn", "explorer", "rl_mt10"):
        for index in (0, 10):
            triple = json.loads((root / f"{key}.json").read_text())[index]
            ex = format_distill(triple["question"], triple["answer"], triple["context"], tok, CFG)
            cfgr.load_state_dict(original)
            cfgr.detector = ReactiveState(detector)
            saved_mask = ctrl.mask.detach().clone()
            mask, _, _ = cfgr(model, ex["recfg"].unsqueeze(0).to("cuda"))
            stats = cfgr.last_stats
            t_ids = ex["t_ids"].unsqueeze(0).to("cuda")
            t_lbl = ex["t_lbl"].unsqueeze(0).to("cuda")
            ctrl.set_mask(torch.zeros(ctrl.H, device=ctrl.mask.device))
            with torch.no_grad():
                teacher, targets = _gather_answer_logits(model(input_ids=t_ids).logits, t_lbl)
                teacher_ce = F.cross_entropy(teacher.float(), targets, reduction="none")
            zero = torch.zeros_like(mask)
            print("\nCASE", key, index, "QUESTION", triple["question"], flush=True)
            print("REFERENCE", repr(triple["answer"][:280]), flush=True)
            print("SEARCH", stats, "mask_overlap_saved", float((saved_mask*mask).sum()/CFG.k), flush=True)
            print("TEACHER CE", round(teacher_ce.mean().item(),3), "first8", round(teacher_ce[:8].mean().item(),3), flush=True)
            for name, m in (("zero", zero), ("saved", saved_mask), ("updated", mask)):
                print("METRICS", name, score(model, ctrl, ex, teacher, m), flush=True)
            for name, m in (("zero", zero), ("updated", mask)):
                for thinking in (False, True):
                    print("GENERATE", name, "thinking", thinking,
                          repr(generate(model, tok, ctrl, m, triple["question"], thinking)[:350]), flush=True)


if __name__ == "__main__":
    main()
