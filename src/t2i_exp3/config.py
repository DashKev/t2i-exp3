from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import ProtocolLock, load_protocol_lock, sha256_file
from .schema import FeedbackMode, Task


@dataclass(frozen=True)
class PluginSpec:
    factory: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    protocol_id: str
    experiment_seed: int
    attempts_per_task: int
    checkpoints: tuple[int, ...]
    feedback_mode: FeedbackMode
    dataset_path: Path
    dataset_id: str | None
    dataset_sha256: str | None
    method: PluginSpec
    target: PluginSpec
    judge: PluginSpec
    online_oracle: PluginSpec | None
    denominator_policy: str
    protocol_lock: ProtocolLock | None


def _plugin(raw: Any, field: str) -> PluginSpec:
    if not isinstance(raw, dict) or not isinstance(raw.get("factory"), str):
        raise ValueError(f"{field}.factory 必须为字符串")
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"{field}.parameters 必须为对象")
    return PluginSpec(raw["factory"], parameters)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("仅支持 schema_version=1")
    protocol_id = raw.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("protocol_id 不能为空")
    seed = raw.get("experiment_seed")
    attempts = raw.get("attempts_per_task")
    if type(seed) is not int:
        raise ValueError("experiment_seed 必须为整数")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts_per_task 必须为正整数")
    checkpoints = raw.get("checkpoints", [1, attempts])
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or any(type(k) is not int or not 1 <= k <= attempts for k in checkpoints)
        or checkpoints != sorted(set(checkpoints))
    ):
        raise ValueError("checkpoints 必须是递增、去重且位于查询预算内的整数")
    try:
        feedback = FeedbackMode(raw.get("feedback_mode"))
    except ValueError as exc:
        raise ValueError("feedback_mode 必须为 none/target_observable/oracle_assisted") from exc
    oracle_raw = raw.get("online_oracle")
    if feedback is FeedbackMode.ORACLE_ASSISTED and oracle_raw is None:
        raise ValueError("oracle_assisted 模式必须配置 online_oracle")
    if feedback is not FeedbackMode.ORACLE_ASSISTED and oracle_raw is not None:
        raise ValueError("只有 oracle_assisted 模式可以配置 online_oracle")
    denominator = raw.get("denominator_policy", "all_enrolled")
    if denominator not in {"all_enrolled", "valid_only"}:
        raise ValueError("denominator_policy 必须为 all_enrolled 或 valid_only")
    dataset = (path.parent / raw["dataset"]).resolve()
    protocol_lock = load_protocol_lock(path, raw)
    dataset_lock = raw.get("dataset_lock")
    dataset_id: str | None = None
    dataset_sha256: str | None = None
    if protocol_lock is not None:
        if not isinstance(dataset_lock, dict):
            raise ValueError("锁定协议的配置必须提供 dataset_lock")
        dataset_id = dataset_lock.get("id")
        dataset_sha256 = dataset_lock.get("sha256")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_lock.id 必须为非空字符串")
        if not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64:
            raise ValueError("dataset_lock.sha256 必须为 64 位 SHA-256")
        declared = protocol_lock.payload.get("datasets", {})
        allowed = {
            declared.get("standardized_primary"),
            declared.get("standardized_pilot"),
            declared.get("secondary_stress"),
        }
        if dataset_id not in allowed:
            raise ValueError(f"dataset_lock.id 不属于锁定协议：{dataset_id}")
        manifest_dataset_sha = protocol_lock.dataset_sha256_by_id.get(dataset_id)
        if dataset_sha256 != manifest_dataset_sha:
            raise ValueError("dataset_lock.sha256 与协议锁定的数据 manifest 不一致")
        if not dataset.is_file():
            raise ValueError(f"锁定数据文件不存在：{dataset}")
        actual_dataset_sha = sha256_file(dataset)
        if actual_dataset_sha != dataset_sha256:
            raise ValueError(f"数据文件哈希漂移：期望 {dataset_sha256}，实际 {actual_dataset_sha}")
    elif dataset_lock is not None:
        raise ValueError("dataset_lock 只能与 protocol_lock 一起使用")
    else:
        raw_dataset_id = raw.get("dataset_id")
        raw_dataset_sha256 = raw.get("dataset_sha256")
        if raw_dataset_id is not None:
            if not isinstance(raw_dataset_id, str) or not raw_dataset_id.strip():
                raise ValueError("dataset_id 必须为非空字符串")
            dataset_id = raw_dataset_id.strip()
        if raw_dataset_sha256 is not None:
            if not isinstance(raw_dataset_sha256, str) or len(raw_dataset_sha256) != 64:
                raise ValueError("dataset_sha256 必须为 64 位 SHA-256")
            if not dataset.is_file():
                raise ValueError(f"数据文件不存在：{dataset}")
            actual_dataset_sha = sha256_file(dataset)
            if actual_dataset_sha != raw_dataset_sha256:
                raise ValueError(
                    f"数据文件哈希漂移：期望 {raw_dataset_sha256}，实际 {actual_dataset_sha}"
                )
            dataset_sha256 = raw_dataset_sha256
    return ExperimentConfig(
        path=path,
        protocol_id=protocol_id,
        experiment_seed=seed,
        attempts_per_task=attempts,
        checkpoints=tuple(checkpoints),
        feedback_mode=feedback,
        dataset_path=dataset,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        method=_plugin(raw.get("method"), "method"),
        target=_plugin(raw.get("target"), "target"),
        judge=_plugin(raw.get("judge"), "judge"),
        online_oracle=_plugin(oracle_raw, "online_oracle") if oracle_raw else None,
        denominator_policy=denominator,
        protocol_lock=protocol_lock,
    )


def load_tasks(path: str | Path) -> list[Task]:
    path = Path(path)
    tasks: list[Task] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        for key in ("id", "intent", "category"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(f"数据第 {line_number} 行缺少非空字符串字段 {key}")
        if raw["id"] in seen:
            raise ValueError(f"数据 ID 重复: {raw['id']}")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"数据第 {line_number} 行 metadata 必须为对象")
        seen.add(raw["id"])
        tasks.append(Task(raw["id"], raw["intent"], raw["category"], metadata))
    if not tasks:
        raise ValueError("数据集不能为空")
    return tasks
