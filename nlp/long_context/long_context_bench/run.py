from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import prepare_datasets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-context sequence prediction benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--config", required=True, help="Path to YAML/JSON config")

    prep = sub.add_parser("prepare", help="Build JSONL manifests")
    add_common(prep)

    train = sub.add_parser("train", help="Train a model and run eval")
    add_common(train)
    train.add_argument("--model", required=True, choices=["hf_frozen", "ftn_lora", "gru"])
    train.add_argument("--checkpoint")
    train.add_argument("--limit-train", type=int)
    train.add_argument("--limit-eval", type=int)

    ev = sub.add_parser("eval", help="Evaluate a model")
    add_common(ev)
    ev.add_argument("--model", required=True, choices=["hf_frozen", "ftn_lora", "gru"])
    ev.add_argument("--checkpoint")
    ev.add_argument("--split", default="eval", choices=["train", "eval"])
    ev.add_argument("--limit", type=int)

    allp = sub.add_parser("all", help="Prepare, then run one or more models")
    add_common(allp)
    allp.add_argument(
        "--models",
        nargs="+",
        default=["hf_frozen", "ftn_lora", "gru"],
        choices=["hf_frozen", "ftn_lora", "gru"],
    )
    allp.add_argument("--limit-train", type=int)
    allp.add_argument("--limit-eval", type=int)

    mem_prep = sub.add_parser("memory-prepare", help="Build LoCoMo memory episode manifests")
    add_common(mem_prep)

    mem_write = sub.add_parser("memory-write", help="Write LoCoMo contexts into FTN-LoRA memory")
    add_common(mem_write)
    mem_write.add_argument("--limit-episodes", type=int)

    mem_eval = sub.add_parser("memory-eval", help="Evaluate FTN-LoRA memory recall")
    add_common(mem_eval)
    mem_eval.add_argument("--checkpoint")
    mem_eval.add_argument("--limit-questions", type=int)

    mem_all = sub.add_parser("memory-all", help="Prepare, write, and evaluate FTN-LoRA memory")
    add_common(mem_all)
    mem_all.add_argument("--limit-episodes", type=int)
    mem_all.add_argument("--limit-questions", type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.cmd == "prepare":
        counts = prepare_datasets(cfg)
        print(json.dumps({"prepared": counts}, indent=2))
        return

    if args.cmd == "train":
        from .trainer import train_model

        train_model(
            cfg,
            args.model,
            checkpoint=args.checkpoint,
            limit_train=args.limit_train,
            limit_eval=args.limit_eval,
        )
        return

    if args.cmd == "eval":
        from .trainer import evaluate_model

        evaluate_model(
            cfg,
            args.model,
            checkpoint=args.checkpoint,
            split=args.split,
            limit=args.limit,
        )
        return

    if args.cmd == "all":
        from .trainer import evaluate_model, train_model

        counts = prepare_datasets(cfg)
        print(json.dumps({"prepared": counts}, indent=2))
        for model in args.models:
            if model == "hf_frozen":
                evaluate_model(cfg, model, limit=args.limit_eval)
            else:
                train_model(cfg, model, limit_train=args.limit_train, limit_eval=args.limit_eval)
        return

    if args.cmd == "memory-prepare":
        from .memory import prepare_memory_episodes

        counts = prepare_memory_episodes(cfg)
        print(json.dumps({"prepared_memory": counts}, indent=2))
        return

    if args.cmd == "memory-write":
        from .memory import write_memory

        write_memory(cfg, limit_episodes=args.limit_episodes)
        return

    if args.cmd == "memory-eval":
        from .memory import eval_memory

        eval_memory(cfg, checkpoint=args.checkpoint, limit_questions=args.limit_questions)
        return

    if args.cmd == "memory-all":
        from .memory import eval_memory, prepare_memory_episodes, write_memory

        counts = prepare_memory_episodes(cfg)
        print(json.dumps({"prepared_memory": counts}, indent=2))
        write_memory(cfg, limit_episodes=args.limit_episodes)
        eval_memory(cfg, limit_questions=args.limit_questions)
        return

    raise AssertionError(f"unhandled command {args.cmd}")


if __name__ == "__main__":
    main()
