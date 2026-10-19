"""Offline answer-aware mask probe; never used by training or serving."""
import json
import math

import torch
import torch.nn.functional as F

torch.set_default_device("cuda")

from config import CFG
from data import format_distill
from model import build, load_bank
from spatial import kwta


def loss_for(model, ctrl, ids, targets, mask):
    ctrl.set_mask(mask)
    out = model(input_ids=ids).logits
    logits = out[:, -targets.numel()-1:-1, :].float().reshape(-1, out.shape[-1])
    return F.cross_entropy(logits, targets)


def generate(model, tok, ctrl, mask, q):
    ctrl.set_mask(mask)
    enc = tok.apply_chat_template(
        [{"role":"system","content":CFG.qa_sys},{"role":"user","content":q}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=72, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,enc["input_ids"].shape[1]:],skip_special_tokens=False)


def main():
    model,tok,ctrl,wrappers,_ = build(CFG)
    load_bank(wrappers,CFG.ckpt_dir/"bank.pt")
    model.gradient_checkpointing_disable()
    model.config.use_cache=True
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    posterior=torch.load(CFG.ckpt_dir/"posterior.pt",map_location="cpu",weights_only=True)
    for key,index in (("base_ftn",0),("rl_mt10",0)):
        triple=json.loads((CFG.data_cache/"qa"/f"{key}.json").read_text())[index]
        ex=format_distill(triple["question"],triple["answer"],triple["context"],tok,CFG)
        labels=ex["lbl"]
        start=int((labels!=-100).nonzero()[0])
        n=min(64,len(labels)-start)
        ids=ex["ids"][:start+n].unsqueeze(0).to("cuda")
        targets=labels[start:start+n].to("cuda")
        prior=posterior["mask_logits"].to("cuda")
        saved=posterior["mask"].to("cuda")
        with torch.no_grad():
            saved_loss=float(loss_for(model,ctrl,ids,targets,saved))
        logits=prior.clone().requires_grad_(True)
        opt=torch.optim.Adam([logits],lr=0.3)
        best=(saved_loss,saved.detach().clone())
        for step in range(25):
            soft=torch.sigmoid(logits/3)
            u=torch.rand_like(logits).clamp_(1e-7,1-1e-7)
            hard=kwta(logits/3-torch.log(-torch.log(u)),ctrl.k)
            objective=loss_for(model,ctrl,ids,targets,hard+soft-soft.detach())
            value=float(objective.detach())
            if value<best[0]:
                best=(value,hard.detach().clone())
            opt.zero_grad()
            objective.backward()
            opt.step()
            if step in (0,9,24):
                print(key,"step",step,"answer_ce64",round(value,3),"best",round(best[0],3),flush=True)
        with torch.no_grad():
            print("\nCASE",key,"QUESTION",triple["question"],flush=True)
            print("REF",repr(triple["answer"][:220]),flush=True)
            for name,mask in (("saved",saved),("oracle",best[1])):
                print(name,"ce64",round(float(loss_for(model,ctrl,ids,targets,mask)),3),
                      "overlap_saved",round(float((saved*mask).sum()/ctrl.k),3),
                      "generated",repr(generate(model,tok,ctrl,mask,triple["question"])[:350]),flush=True)


if __name__=="__main__":
    main()
