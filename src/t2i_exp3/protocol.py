from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProtocolLock:
    path: Path
    sha256: str
    payload: dict[str, Any]
    dataset_sha256_by_id: dict[str, str]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol_lock(config_path: Path, raw: dict[str, Any]) -> ProtocolLock | None:
    reference = raw.get("protocol_lock")
    if reference is None:
        return None
    if not isinstance(reference, dict):
        raise ValueError("protocol_lock 必须为对象")
    relative = reference.get("path")
    expected = reference.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError("protocol_lock.path 必须为非空字符串")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("protocol_lock.sha256 必须为 64 位 SHA-256")
    path = (config_path.parent / relative).resolve()
    if not path.is_file():
        raise ValueError(f"协议文件不存在：{path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"协议文件哈希漂移：期望 {expected}，实际 {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError("正式协议必须标记为 frozen")
    if payload.get("protocol_id") != raw.get("protocol_id"):
        raise ValueError("配置 protocol_id 与锁定协议不一致")
    contract = payload.get("query_contract", {})
    if raw.get("attempts_per_task") != contract.get("primary_budget_k"):
        raise ValueError("配置查询预算与锁定协议不一致")
    if raw.get("checkpoints") != contract.get("checkpoints"):
        raise ValueError("配置 checkpoints 与锁定协议不一致")
    if raw.get("denominator_policy", "all_enrolled") != payload.get("outcome_contract", {}).get("denominator"):
        raise ValueError("配置分母策略与锁定协议不一致")
    allowed_feedback = payload.get("threat_model", {}).get("separate_tracks", [])
    if raw.get("feedback_mode") not in allowed_feedback:
        raise ValueError("配置 feedback_mode 不属于锁定协议的结果轨道")
    datasets = payload.get("datasets", {})
    manifest_relative = datasets.get("manifest")
    manifest_expected = datasets.get("manifest_sha256")
    if not isinstance(manifest_relative, str) or not isinstance(manifest_expected, str):
        raise ValueError("锁定协议缺少数据 manifest 或其 SHA-256")
    manifest_path = next(
        (ancestor / manifest_relative for ancestor in path.parents if (ancestor / manifest_relative).is_file()),
        None,
    )
    if manifest_path is None:
        raise ValueError(f"找不到协议声明的数据 manifest：{manifest_relative}")
    if sha256_file(manifest_path) != manifest_expected:
        raise ValueError("协议声明的数据 manifest 已漂移")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_hashes = {
        row["id"]: row["prepared_sha256"]
        for row in manifest.get("datasets", [])
        if isinstance(row.get("id"), str) and isinstance(row.get("prepared_sha256"), str)
    }
    return ProtocolLock(path=path, sha256=actual, payload=payload, dataset_sha256_by_id=dataset_hashes)
