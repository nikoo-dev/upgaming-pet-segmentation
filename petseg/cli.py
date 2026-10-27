"""Run python -m petseg --help for the complete workflow."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .utils import environment, read_json, sha256, write_json


def parser():
    p = argparse.ArgumentParser(description="Pet foreground segmentation: reproducible experiments and local demo")
    p.add_argument("--threads", type=int, default=1, help="Fixed torch CPU thread count")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("env")
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", default="data")
    prep.add_argument("--output", default="artifacts/data")
    prep.add_argument("--download", action="store_true")
    tr = sub.add_parser("train")
    tr.add_argument("--config", required=True)
    tr.add_argument("--root", default="data")
    tr.add_argument("--manifest", default="artifacts/data/splits.json")
    tr.add_argument("--output", required=True)
    tr.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    tr.add_argument("--resume")
    sel = sub.add_parser("select")
    sel.add_argument("--runs", default="artifacts/runs")
    sel.add_argument("--manifest", default="artifacts/data/splits.json")
    sel.add_argument("--output", default="artifacts/final")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--root", default="data")
    bench.add_argument("--manifest", default="artifacts/data/splits.json")
    bench.add_argument("--checkpoint", default="artifacts/final/selected.pt")
    bench.add_argument("--selection", default="artifacts/final/selection.json")
    bench.add_argument("--output", default="artifacts/final/decision.json")
    bench.add_argument("--calls", type=int, default=100)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--root", default="data")
    ev.add_argument("--manifest", default="artifacts/data/splits.json")
    ev.add_argument("--checkpoint", default="artifacts/final/selected.pt")
    ev.add_argument("--split", choices=["val", "test"], default="val")
    ev.add_argument("--size", type=int, default=256)
    ev.add_argument("--decision", default="artifacts/final/decision.json")
    ev.add_argument("--output", help="Default: artifacts/evaluations/<split>_<size>")
    pred = sub.add_parser("predict")
    pred.add_argument("image")
    pred.add_argument("--checkpoint", default="artifacts/final/selected.pt")
    pred.add_argument("--decision", default="artifacts/final/decision.json")
    pred.add_argument("--size", type=int, help="Default: measured deployment choice; explicit size is a manual override")
    pred.add_argument("--output", default="artifacts/prediction")
    rep = sub.add_parser("report")
    rep.add_argument("--artifacts", default="artifacts")
    return p


def main():
    args = parser().parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.hub.set_dir(str(Path("artifacts/cache/torch").resolve()))
    if args.command == "env":
        print(json.dumps(environment(), indent=2))
    elif args.command == "prepare":
        from .data import prepare
        result = prepare(args.root, args.output, args.download)
        print(json.dumps(result["counts"], indent=2))
    elif args.command == "train":
        from .train import load_config, train
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU is available. Select a GPU Colab runtime, or use --device cpu and configs/cpu_*.json")
        train(args.root, args.manifest, args.output, load_config(args.config), args.device, args.resume)
    elif args.command == "select":
        from .benchmark import select_checkpoint
        print(json.dumps(select_checkpoint(args.runs, args.manifest, args.output), indent=2))
    elif args.command == "benchmark":
        from .benchmark import benchmark
        result = benchmark(args.root, args.manifest, args.checkpoint, args.selection, args.output, args.threads, args.calls)
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        from .inference import load_model, validate_deployment
        from .evaluate import evaluate_model, save_evaluation
        model, state = load_model(args.checkpoint)
        if args.split == "test":
            decision = read_json(args.decision)
            deployment_size = validate_deployment(decision, sha256(args.checkpoint), state, sha256(args.manifest))
            if args.size not in [256, deployment_size]:
                raise ValueError("Final test evaluation allows only the frozen baseline/deployment resolutions")
        if state["manifest_sha256"] != sha256(args.manifest):
            raise ValueError("Evaluation splits differ from training")
        if args.split == "test" and state["config"]["diagnostic"]:
            raise ValueError("Do not use final test for diagnostic runs")
        records = read_json(args.manifest)["splits"][args.split]
        metrics, rows = evaluate_model(model, args.root, records, args.size)
        metrics.update({"split": args.split, "checkpoint_sha256": sha256(args.checkpoint),
                        "manifest_sha256": sha256(args.manifest), "boundary_policy": "raw label 3 ignored"})
        if args.split == "test":
            metrics["decision_sha256"] = sha256(args.decision)
        output = args.output or f"artifacts/evaluations/{args.split}_{args.size}"
        save_evaluation(output, metrics, rows, model, args.root, records)
        print(json.dumps(metrics, indent=2))
    elif args.command == "predict":
        from .inference import load_upload, predict
        decision = read_json(args.decision) if args.size is None else None
        result = predict(load_upload(Path(args.image).read_bytes()), args.checkpoint,
                         args.size if args.size is not None else 256, decision=decision)
        if decision is not None:
            result["metadata"]["decision_sha256"] = sha256(args.decision)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result["mask"] * 255).save(output / "mask.png")
        result["overlay"].save(output / "overlay.png")
        result["composite"].save(output / "composite.png")
        write_json(output / "metadata.json", result["metadata"])
        print(f"Saved outputs to {output.resolve()}")
    elif args.command == "report":
        from .report import build_report
        print(json.dumps(build_report(args.artifacts), indent=2))


if __name__ == "__main__":
    main()
