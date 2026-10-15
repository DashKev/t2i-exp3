"""Render local settings without importing private configuration."""
import argparse
import copy
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, default=Path("configs/sd35.example.json"))
    p.add_argument("--output", type=Path, default=Path("configs/sd35.local.json"))
    p.add_argument("--dataset", type=Path, default=Path("data/prepared/i2p200.jsonl"))
    p.add_argument("--method", choices=["exp3", "static5", "original"], default="exp3")
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--llm-base-url", default="http://127.0.0.1:18080")
    p.add_argument("--llm-model", default="local-mutator")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--guard-device", default="cuda:1")
    p.add_argument("--judge-device", default="cuda:1")
    p.add_argument("--cpu-offload", action="store_true")
    a = p.parse_args()
    if a.budget < 1:
        p.error("budget must be positive")
    cfg = copy.deepcopy(json.loads(a.template.read_text()))
    cfg["dataset"] = str(a.dataset.resolve())
    k = 1 if a.method == "original" else a.budget
    cfg["attempts_per_task"] = k
    cfg["checkpoints"] = sorted(set([i for i in (1, 2, 4, 8, 16) if i <= k] + [k]))
    cfg["method"]["parameters"].update(maximum_queries=k, base_url=a.llm_base_url, model=a.llm_model)
    cfg["method"]["factory"] = {"exp3": "t2i_exp3.methods.feedback_routing_adapter:build",
        "static5": "t2i_exp3.methods.static:build", "original": "t2i_exp3.methods.original:build"}[a.method]
    target = cfg["target"]["parameters"]
    target.update(device=a.device, cpu_offload=a.cpu_offload, device_map=None)
    target["input_guard"]["device"] = a.guard_device
    checker = target["output_checker"]
    for c in checker.get("checkers", [checker]):
        c["device"] = a.judge_device
    cfg["online_oracle"]["parameters"]["device"] = a.judge_device
    cfg["judge"]["parameters"]["device"] = a.judge_device
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(a.output)
