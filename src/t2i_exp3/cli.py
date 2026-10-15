from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import resume_experiment, run_experiment, validate_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="t2i-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="校验配置和数据，不执行目标查询")
    validate.add_argument("config")
    run = sub.add_parser("run", help="执行实验")
    run.add_argument("config")
    run.add_argument("--output", default="results")
    run.add_argument("--task-id", action="append", default=None)
    run.add_argument("--task-shard-index", type=int, default=None)
    run.add_argument("--task-shards", type=int, default=None)
    resume = sub.add_parser("resume", help="从未完成 run 目录恢复实验")
    resume.add_argument("run_dir")
    resume.add_argument(
        "--config",
        default=None,
        help=(
            "用指定配置恢复既有 run。用于只改变执行拓扑/设备映射的场景；"
            "协议、数据与任务选择仍会按原 manifest 校验。"
        ),
    )
    resume.add_argument(
        "--allow-config-drift",
        action="store_true",
        help="显式允许 config 文件哈希不同；数据、协议和任务选择仍必须一致",
    )
    resume.add_argument(
        "--extend-budget",
        action="store_true",
        help=(
            "允许用更大 attempts_per_task/checkpoints 的配置扩展已完成 run；"
            "旧 metrics/completion/manifest 会先归档，再在同一 run_dir 继续追加查询"
        ),
    )
    report = sub.add_parser("report", help="输出已有运行的指标")
    report.add_argument("run_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        config, task_count = validate_experiment(args.config)
        print(json.dumps({
            "valid": True,
            "protocol_id": config.protocol_id,
            "feedback_mode": config.feedback_mode.value,
            "tasks": task_count,
            "planned_target_queries": task_count * config.attempts_per_task,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        directory = run_experiment(
            args.config,
            args.output,
            task_ids=args.task_id,
            task_shard_index=args.task_shard_index,
            task_shards=args.task_shards,
        )
        print(directory)
        return 0
    if args.command == "resume":
        directory = resume_experiment(
            args.run_dir,
            allow_config_drift=args.allow_config_drift,
            config_override_path=args.config,
            extend_budget=args.extend_budget,
        )
        print(directory)
        return 0
    metrics = json.loads((Path(args.run_dir) / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
