"""Read-only probe of chat templates and archived input routing on the pod."""
import json
import torch

from config import CFG
from data import load_qa
from input_routing import InputRoutingArchive
from model import build, load_bank
from spatial import kwta


def main():
    torch.set_default_device("cuda")
    model, tok, ctrl, wrappers, _ = build(CFG)
    load_bank(wrappers, CFG.ckpt_dir / "bank.pt")
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    router = InputRoutingArchive(CFG, tok, model)
    router.load_state_dict(torch.load(CFG.ckpt_dir / "routing.pt", map_location="cpu", weights_only=True))
    questions = ["What is the purpose of the `reconfigure` method in the FTN class?",
                 "What is FTN?", "How does reconfiguration work in base FTN?",
                 "What is RDDPG_dual?"]
    questions += [load_qa(CFG, "base_ftn")[0]["question"]]
    for question in questions:
        mask, info = router.match(question)
        print("ROUTE", json.dumps({"question": question, **info}), flush=True)
    question = questions[0]
    ctrl.set_mask(kwta(router.routes["base_ftn"]["logits"].to("cuda"), CFG.k))
    msgs = [{"role": "system", "content": CFG.qa_sys}, {"role": "user", "content": question}]
    for mode in ("current_off", "template_off", "template_on"):
        options = {} if mode == "current_off" else {"enable_thinking": mode == "template_on"}
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True, **options).to("cuda")
        print("PREFIX", mode, repr(tok.decode(enc["input_ids"][0][-35:], skip_special_tokens=False)), flush=True)
        kwargs = {}
        if mode == "current_off":
            kwargs["bad_words_ids"] = [[tok.convert_tokens_to_ids(t)] for t in ("<think>", "</think>")]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tok.eos_token_id, **kwargs)
        print("ANSWER", mode, repr(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)), flush=True)


if __name__ == "__main__":
    main()
