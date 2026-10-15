from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any
from uuid import uuid4

from .artifacts import append_jsonl, read_jsonl, sha256_file, write_json_once
from .config import ExperimentConfig, load_config, load_tasks
from .metrics import compute_metrics
from .plugins import RuntimeContext, build_plugin
from .schema import Assessment, Candidate, FeedbackMode, PolicyEvent, TargetObservation, TargetStatus, jsonable


FATAL_INFRASTRUCTURE_ERROR_SNIPPETS = (
    "CUDA error: an illegal memory access",
    "device-side assert triggered",
    "unspecified launch failure",
    "CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH",
    "CUBLAS_STATUS_EXECUTION_FAILED",
    "CUDA out of memory",
    "OutOfMemoryError",
)


class FatalTargetInfrastructureError(RuntimeError):
    """Non-recoverable target-side failure that would poison the remaining run.

    These errors are different from ordinary blocked/refused target feedback or
    transient API errors. Continuing after a CUDA illegal-memory/cudnn stream
    mismatch typically turns every following attempt into an infrastructure
    error, producing a misleading "0%" result table. Failing fast preserves
    benchmark validity and lets the orchestration layer restart with a clean
    topology/config.
    """


@dataclass(frozen=True)
class _Components:
    method: Any
    target: Any
    judge: Any
    online_oracle: Any | None
    method_query_limit: int | None


def _root() -> Path:
    # Working-directory contract works for editable and wheel installations.
    return Path.cwd().resolve()


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _derive_seed(experiment_seed: int, task_id: str, query_index: int) -> int:
    raw = f"{experiment_seed}:{task_id}:{query_index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _attempt_id(task_id: str, query_index: int) -> str:
    return hashlib.sha256(f"{task_id}:{query_index}".encode()).hexdigest()[:24]


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _contains_fatal_infrastructure_error(text: str) -> bool:
    lowered = text.lower()
    return any(snippet.lower() in lowered for snippet in FATAL_INFRASTRUCTURE_ERROR_SNIPPETS)


def _fatal_infrastructure_message(
    *,
    exc: BaseException | None = None,
    observation: TargetObservation | None = None,
) -> str | None:
    parts: list[str] = []
    if exc is not None:
        parts.append(type(exc).__name__)
        parts.append(str(exc))
    if observation is not None:
        parts.append(str(observation.status.value))
        for value in observation.private_metadata.values():
            if isinstance(value, (str, int, float, bool)):
                parts.append(str(value))
    text = "\n".join(parts)
    return text if _contains_fatal_infrastructure_error(text) else None


def _new_run_dir(output_root: Path, protocol_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in protocol_id)
    directory = output_root / f"{stamp}-{safe}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    (directory / "artifacts").mkdir(mode=0o700)
    return directory


def _select_task_shard(tasks: list[Any], index: int | None, shards: int | None) -> list[Any]:
    if index is None and shards is None:
        return tasks
    if index is None or shards is None:
        raise ValueError("task_shard_index 和 task_shards 必须同时提供")
    if shards < 1:
        raise ValueError("task_shards 必须为正整数")
    if index < 0 or index >= shards:
        raise ValueError("task_shard_index 必须位于 [0, task_shards)")
    return [task for position, task in enumerate(tasks) if position % shards == index]


def _select_task_ids(tasks: list[Any], task_ids: list[str] | None) -> list[Any]:
    if not task_ids:
        return tasks
    wanted = [str(task_id) for task_id in task_ids]
    wanted_set = set(wanted)
    selected = [task for task in tasks if task.id in wanted_set]
    found = {task.id for task in selected}
    missing = [task_id for task_id in wanted if task_id not in found]
    if missing:
        raise ValueError(f"锁定数据集中找不到指定 task_id：{missing}")
    return selected


def validate_experiment(config_path: str | Path) -> tuple[ExperimentConfig, int]:
    config = load_config(config_path)
    tasks = load_tasks(config.dataset_path)
    return config, len(tasks)


def _apply_outcome_contract(
    config: ExperimentConfig,
    observation: TargetObservation,
    assessment: Assessment,
) -> Assessment:
    if config.protocol_lock is None:
        return assessment
    outcome = config.protocol_lock.payload["outcome_contract"]
    threshold = float(outcome["semantic_evaluator"]["threshold"])
    valid_artifact = observation.status is TargetStatus.RETURNED and bool(observation.artifact_paths)
    policy_violation = assessment.policy_violation
    semantic_score = assessment.semantic_score
    if not valid_artifact:
        success = False
    elif policy_violation is False:
        success = False
    elif policy_violation is None or semantic_score is None:
        success = None
    else:
        success = semantic_score >= threshold
    return Assessment(
        success=success,
        bypassed=valid_artifact,
        policy_violation=policy_violation,
        semantic_score=semantic_score,
        labels=assessment.labels,
        details={
            **assessment.details,
            "outcome_contract_applied": config.protocol_lock.payload["protocol_id"],
            "valid_artifact": valid_artifact,
            "semantic_threshold": threshold,
        },
    )


def _component_usage(component: Any) -> dict[str, Any]:
    snapshot = getattr(component, "usage_snapshot", None)
    if snapshot is None:
        return {}
    if not callable(snapshot):
        raise TypeError("usage_snapshot 必须可调用")
    usage = snapshot()
    if not isinstance(usage, dict):
        raise TypeError("usage_snapshot 必须返回对象")
    return jsonable(usage)


def _build_components(config: ExperimentConfig, context: RuntimeContext) -> _Components:
    method = build_plugin(config.method, context)
    target = build_plugin(config.target, context)
    judge = build_plugin(config.judge, context)
    online_oracle = build_plugin(config.online_oracle, context) if config.online_oracle else None
    method_query_limit = getattr(method, "maximum_queries", None)
    if callable(method_query_limit):
        method_query_limit = method_query_limit()
    if method_query_limit is not None:
        method_query_limit = int(method_query_limit)
        if method_query_limit < 1:
            raise ValueError("method.maximum_queries 必须为正整数")
    return _Components(method, target, judge, online_oracle, method_query_limit)


def _manifest_for_new_run(
    *,
    root: Path,
    config: ExperimentConfig,
    full_tasks: list[Any],
    tasks: list[Any],
    task_ids: list[str] | None,
    task_shard_index: int | None,
    task_shards: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config.protocol_id,
        "feedback_mode": config.feedback_mode.value,
        "attempts_per_task": config.attempts_per_task,
        "checkpoints": list(config.checkpoints),
        "denominator_policy": config.denominator_policy,
        "enrolled_tasks": len(tasks),
        "planned_target_queries": len(tasks) * config.attempts_per_task,
        "config_path": str(config.path),
        "config_sha256": sha256_file(config.path),
        "dataset_path": str(config.dataset_path),
        "dataset_id": config.dataset_id,
        "dataset_sha256": sha256_file(config.dataset_path),
        "expected_dataset_sha256": config.dataset_sha256,
        "full_dataset_tasks": len(full_tasks),
        "task_ids": list(task_ids) if task_ids else None,
        "task_shard": (
            {"index": task_shard_index, "shards": task_shards, "selection": "position_modulo"}
            if task_shard_index is not None
            else None
        ),
        "protocol_lock_path": str(config.protocol_lock.path) if config.protocol_lock else None,
        "protocol_lock_sha256": config.protocol_lock.sha256 if config.protocol_lock else None,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_status": _git(root, "status", "--porcelain"),
        "git_submodules": _git(root, "submodule", "status", "--recursive"),
        "python": platform.python_version(),
        "package_versions": _package_versions(
            (
                "torch",
                "diffusers",
                "transformers",
                "Pillow",
                "numpy",
                "onnxruntime",
                "opencv-python-headless",
            )
        ),
        "platform": platform.platform(),
        "argv": sys.argv,
    }


def _validate_observation_artifacts(
    run_dir: Path,
    observation: TargetObservation,
) -> tuple[TargetObservation, list[dict[str, Any]]]:
    artifact_records = []
    try:
        for relative in observation.artifact_paths:
            artifact = (run_dir / relative).resolve()
            if not artifact.is_relative_to(run_dir.resolve()) or not artifact.is_file():
                raise ValueError(f"目标返回了无效产物路径：{relative}")
            artifact_records.append({
                "path": relative,
                "sha256": sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
            })
    except Exception as exc:
        observation = TargetObservation(
            status=TargetStatus.INFRASTRUCTURE_ERROR,
            public_feedback={"result": "unavailable"},
            private_metadata={"artifact_error_type": type(exc).__name__, "artifact_error": str(exc)},
        )
        artifact_records = []
    return observation, artifact_records


def _generate_attempt_row(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    target: Any,
    task: Any,
    candidate: Candidate,
    query_index: int,
) -> tuple[dict[str, Any], TargetObservation]:
    generation_seed = _derive_seed(config.experiment_seed, task.id, query_index)
    try:
        observation = target.generate(task, candidate, generation_seed, run_dir / "artifacts")
        if not isinstance(observation, TargetObservation):
            raise TypeError("target.generate 必须返回 TargetObservation")
    except Exception as exc:
        fatal_message = _fatal_infrastructure_message(exc=exc)
        if fatal_message is not None:
            raise FatalTargetInfrastructureError(fatal_message) from exc
        observation = TargetObservation(
            status=TargetStatus.INFRASTRUCTURE_ERROR,
            public_feedback={"result": "unavailable"},
            private_metadata={"exception_type": type(exc).__name__, "exception": str(exc)},
        )
    observation, artifact_records = _validate_observation_artifacts(run_dir, observation)
    if observation.status is TargetStatus.INFRASTRUCTURE_ERROR:
        fatal_message = _fatal_infrastructure_message(observation=observation)
        if fatal_message is not None:
            raise FatalTargetInfrastructureError(fatal_message)
    row = {
        "attempt_id": _attempt_id(task.id, query_index),
        "task_id": task.id,
        "category": task.category,
        "query_index": query_index,
        "generation_seed": generation_seed,
        "candidate": jsonable(candidate),
        "observation": jsonable(observation),
        "artifacts": artifact_records,
    }
    return row, observation


def _online_feedback_record(
    row: dict[str, Any],
    oracle_feedback: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    prompt = str(row.get("candidate", {}).get("prompt", ""))
    return {
        "attempt_id": row["attempt_id"],
        "task_id": row["task_id"],
        "query_index": row["query_index"],
        "generation_seed": row["generation_seed"],
        "candidate_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "oracle_feedback": jsonable(dict(oracle_feedback)),
        "source": source,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }


def _append_online_feedback(
    run_dir: Path,
    row: dict[str, Any],
    oracle_feedback: dict[str, Any],
    *,
    source: str,
) -> None:
    append_jsonl(run_dir / "online_feedback.jsonl", _online_feedback_record(row, oracle_feedback, source=source))


def _event_from_row(
    config: ExperimentConfig,
    row: dict[str, Any],
    candidate: Candidate,
    observation: TargetObservation,
    oracle_feedback: dict[str, Any] | None,
) -> PolicyEvent | None:
    query_index = int(row["query_index"])
    if config.feedback_mode is FeedbackMode.NONE:
        return None
    if config.feedback_mode is FeedbackMode.TARGET_OBSERVABLE:
        return PolicyEvent(query_index, candidate.prompt, observation.policy_view())
    if oracle_feedback is None:
        raise RuntimeError(f"缺少 attempt_id={row['attempt_id']} 的 online oracle feedback")
    return PolicyEvent(query_index, candidate.prompt, observation.policy_view(), dict(oracle_feedback))


def _candidate_from_row(row: dict[str, Any]) -> Candidate:
    raw = row.get("candidate")
    if not isinstance(raw, dict):
        raise ValueError(f"attempt_id={row.get('attempt_id')} 缺少 candidate 对象")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"attempt_id={row.get('attempt_id')} 的 candidate.prompt 为空")
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"attempt_id={row.get('attempt_id')} 的 candidate.metadata 必须为对象")
    return Candidate(prompt, dict(metadata))


def _observation_from_row(row: dict[str, Any]) -> TargetObservation:
    raw = row.get("observation")
    if not isinstance(raw, dict):
        raise ValueError(f"attempt_id={row.get('attempt_id')} 缺少 observation 对象")
    artifact_paths = raw.get("artifact_paths") or ()
    if not isinstance(artifact_paths, (list, tuple)):
        raise ValueError(f"attempt_id={row.get('attempt_id')} 的 artifact_paths 必须为数组")
    public_feedback = raw.get("public_feedback") or {}
    private_metadata = raw.get("private_metadata") or {}
    if not isinstance(public_feedback, dict) or not isinstance(private_metadata, dict):
        raise ValueError(f"attempt_id={row.get('attempt_id')} 的 feedback/metadata 必须为对象")
    return TargetObservation(
        status=TargetStatus(raw["status"]),
        artifact_paths=tuple(str(path) for path in artifact_paths),
        public_feedback=dict(public_feedback),
        private_metadata=dict(private_metadata),
    )


def _load_online_feedback(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "online_feedback.jsonl"
    if not path.exists():
        return {}
    by_attempt: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(read_jsonl(path), 1):
        attempt_id = str(row.get("attempt_id", "")).strip()
        feedback = row.get("oracle_feedback")
        if not attempt_id or not isinstance(feedback, dict):
            raise ValueError(f"{path} 第 {line_number} 行缺少有效 attempt_id/oracle_feedback")
        existing = by_attempt.get(attempt_id)
        if existing is not None and existing != feedback:
            raise ValueError(f"{path} 存在同一 attempt_id 的不一致 online feedback：{attempt_id}")
        by_attempt[attempt_id] = dict(feedback)
    return by_attempt


def _index_attempts(run_dir: Path, config: ExperimentConfig) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    path = run_dir / "attempts.jsonl"
    attempts = read_jsonl(path) if path.exists() else []
    by_task: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for line_number, row in enumerate(attempts, 1):
        task_id = str(row.get("task_id", "")).strip()
        query_index = row.get("query_index")
        if not task_id or type(query_index) is not int or query_index < 1:
            raise ValueError(f"{path} 第 {line_number} 行缺少有效 task_id/query_index")
        key = (task_id, query_index)
        if key in seen:
            raise ValueError(f"{path} 存在重复 task/query：{task_id}#{query_index}")
        seen.add(key)
        expected_attempt_id = _attempt_id(task_id, query_index)
        if row.get("attempt_id") != expected_attempt_id:
            raise ValueError(
                f"{path} 第 {line_number} 行 attempt_id 不匹配："
                f"期望 {expected_attempt_id}，实际 {row.get('attempt_id')}"
            )
        expected_seed = _derive_seed(config.experiment_seed, task_id, query_index)
        if row.get("generation_seed") != expected_seed:
            raise ValueError(
                f"{path} 第 {line_number} 行 generation_seed 不匹配："
                f"期望 {expected_seed}，实际 {row.get('generation_seed')}"
            )
        by_task.setdefault(task_id, []).append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda item: int(item["query_index"]))
        expected = 1
        for row in task_rows:
            if int(row["query_index"]) != expected:
                raise ValueError(
                    f"{path} 中 task_id={row['task_id']} 的 query_index 不连续，"
                    f"期望 {expected}，实际 {row['query_index']}"
                )
            expected += 1
    return attempts, by_task


def _load_assessments(run_dir: Path) -> tuple[list[dict[str, Any]], set[str]]:
    path = run_dir / "assessments.jsonl"
    if not path.exists():
        return [], set()
    assessments = read_jsonl(path)
    assessed_ids: set[str] = set()
    for line_number, row in enumerate(assessments, 1):
        attempt_id = str(row.get("attempt_id", "")).strip()
        if not attempt_id or not isinstance(row.get("assessment"), dict):
            raise ValueError(f"{path} 第 {line_number} 行缺少有效 attempt_id/assessment")
        if attempt_id in assessed_ids:
            raise ValueError(f"{path} 存在重复 assessment：{attempt_id}")
        assessed_ids.add(attempt_id)
    return assessments, assessed_ids


def _tasks_from_manifest(config: ExperimentConfig, manifest: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    full_tasks = load_tasks(config.dataset_path)
    tasks = _select_task_ids(full_tasks, manifest.get("task_ids"))
    shard = manifest.get("task_shard")
    if shard is not None:
        if not isinstance(shard, dict) or shard.get("selection") != "position_modulo":
            raise ValueError("manifest.task_shard 不是支持的 position_modulo 格式")
        tasks = _select_task_shard(tasks, int(shard["index"]), int(shard["shards"]))
    return full_tasks, tasks


def _budget_extension_config_is_compatible(run_dir: Path, config: ExperimentConfig) -> bool:
    snapshot_path = run_dir / "config.snapshot.json"
    if not snapshot_path.is_file():
        return False
    try:
        old_raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
        new_raw = json.loads(config.path.read_text(encoding="utf-8"))
    except Exception:
        return False

    def normalized_budget_source(raw: Any) -> Any:
        clone = json.loads(json.dumps(raw, ensure_ascii=False))
        if not isinstance(clone, dict):
            return clone
        clone.pop("attempts_per_task", None)
        clone.pop("checkpoints", None)
        method = clone.get("method")
        if isinstance(method, dict):
            parameters = method.get("parameters")
            if isinstance(parameters, dict):
                parameters.pop("maximum_queries", None)
        return clone

    return normalized_budget_source(old_raw) == normalized_budget_source(new_raw)


def _validate_resume_manifest(
    run_dir: Path,
    config: ExperimentConfig,
    manifest: dict[str, Any],
    *,
    allow_config_drift: bool,
    extend_budget: bool,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    if manifest.get("schema_version") != 1:
        raise ValueError("当前只支持 schema_version=1 的 run manifest")
    if manifest.get("protocol_id") != config.protocol_id:
        raise ValueError("manifest.protocol_id 与当前配置不一致")
    if manifest.get("feedback_mode") != config.feedback_mode.value:
        raise ValueError("manifest.feedback_mode 与当前配置不一致")
    manifest_attempts = int(manifest.get("attempts_per_task") or 0)
    budget_extension = extend_budget and config.attempts_per_task > manifest_attempts
    if not extend_budget and manifest_attempts != config.attempts_per_task:
        raise ValueError("manifest.attempts_per_task 与当前配置不一致")
    if extend_budget:
        if config.attempts_per_task < manifest_attempts:
            raise ValueError("--extend-budget 只允许扩大 attempts_per_task，不能缩小预算")
        manifest_checkpoints = tuple(int(k) for k in manifest.get("checkpoints", ()))
        if budget_extension:
            expected_old_checkpoints = tuple(
                k for k in config.checkpoints if k <= manifest_attempts
            )
            if manifest_checkpoints != expected_old_checkpoints:
                raise ValueError(
                    "manifest.checkpoints 必须等于新配置中不超过旧预算的 checkpoints，"
                    "才能做可审计预算扩展"
                )
        elif manifest_checkpoints != tuple(config.checkpoints):
            raise ValueError("manifest.checkpoints 与当前配置不一致")
    elif tuple(manifest.get("checkpoints", ())) != tuple(config.checkpoints):
        raise ValueError("manifest.checkpoints 与当前配置不一致")
    if manifest.get("denominator_policy") != config.denominator_policy:
        raise ValueError("manifest.denominator_policy 与当前配置不一致")

    config_sha = sha256_file(config.path)
    compatible_budget_extension_config = (
        budget_extension and _budget_extension_config_is_compatible(run_dir, config)
    )
    if (
        manifest.get("config_sha256") != config_sha
        and not allow_config_drift
        and not compatible_budget_extension_config
    ):
        raise ValueError(
            "配置文件哈希与原 run manifest 不一致；如果确认只是无关变更，"
            "可用 --allow-config-drift 显式恢复"
        )
    dataset_sha = sha256_file(config.dataset_path)
    if manifest.get("dataset_sha256") != dataset_sha:
        raise ValueError("数据文件哈希与原 run manifest 不一致，不能 resume")

    full_tasks, tasks = _tasks_from_manifest(config, manifest)
    if manifest.get("full_dataset_tasks") != len(full_tasks):
        raise ValueError("manifest.full_dataset_tasks 与当前数据不一致")
    if manifest.get("enrolled_tasks") != len(tasks):
        raise ValueError("manifest.enrolled_tasks 与当前任务选择不一致")
    if not tasks:
        raise ValueError("manifest 选择出的任务为空")

    active_manifest = dict(manifest)
    archived_files: dict[str, str] = {}
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        if not budget_extension:
            raise ValueError("该 run 已存在 metrics.json；不能把已完成/已评测 run 当作中断 run resume")
        archived_metrics = run_dir / f"metrics.budget_{manifest_attempts}_completed.{suffix}.json"
        metrics_path.replace(archived_metrics)
        archived_files["metrics"] = str(archived_metrics)
    completion_path = run_dir / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("completed") is True:
            if not budget_extension:
                raise ValueError(f"该 run 已存在 completion.json，当前 completed={completion.get('completed')!r}")
            archived_completion = run_dir / f"completion.budget_{manifest_attempts}_completed.{suffix}.json"
        else:
            archived_completion = run_dir / f"completion.incomplete.{suffix}.json"
        completion_path.replace(archived_completion)
        archived_files["completion"] = str(archived_completion)
    if budget_extension:
        manifest_path = run_dir / "manifest.json"
        archived_manifest = run_dir / f"manifest.budget_{manifest_attempts}.{suffix}.json"
        manifest_path.replace(archived_manifest)
        archived_files["manifest"] = str(archived_manifest)
        active_manifest.update(
            {
                "attempts_per_task": config.attempts_per_task,
                "checkpoints": list(config.checkpoints),
                "planned_target_queries": len(tasks) * config.attempts_per_task,
                "config_path": str(config.path),
                "config_sha256": config_sha,
                "budget_extension": {
                    "previous_attempts_per_task": manifest_attempts,
                    "new_attempts_per_task": config.attempts_per_task,
                    "previous_checkpoints": list(manifest.get("checkpoints", ())),
                    "new_checkpoints": list(config.checkpoints),
                    "config_diff_scope": (
                        "budget_only"
                        if compatible_budget_extension_config and not allow_config_drift
                        else "explicit_allow_config_drift"
                    ),
                    "previous_manifest_path": str(archived_manifest),
                    "extended_utc": datetime.now(timezone.utc).isoformat(),
                    "archived_files": archived_files,
                },
            }
        )
        manifest_path.write_text(
            json.dumps(active_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return full_tasks, tasks, active_manifest


def _record_resume_event(run_dir: Path, payload: dict[str, Any]) -> None:
    append_jsonl(
        run_dir / "resume_events.jsonl",
        {
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            **jsonable(payload),
        },
    )


def _replay_existing_attempt(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    method: Any,
    online_oracle: Any | None,
    task: Any,
    row: dict[str, Any],
    history: list[PolicyEvent],
    online_feedback: dict[str, dict[str, Any]],
) -> tuple[Candidate, TargetObservation, PolicyEvent | None, bool]:
    query_index = int(row["query_index"])
    candidate = _candidate_from_row(row)
    observation = _observation_from_row(row)

    replay_proposal = getattr(method, "replay_proposal", None)
    if replay_proposal is not None:
        replay_proposal(task, tuple(history), query_index, candidate, Random(int(row["generation_seed"])))

    oracle = None
    reconstructed_feedback = False
    if config.feedback_mode is FeedbackMode.ORACLE_ASSISTED:
        if online_oracle is None:
            raise RuntimeError("oracle_assisted resume 缺少 online_oracle")
        oracle = online_feedback.get(row["attempt_id"])
        if oracle is None:
            oracle = dict(online_oracle.feedback(task, candidate, observation))
            online_feedback[row["attempt_id"]] = oracle
            reconstructed_feedback = True
            _append_online_feedback(run_dir, row, oracle, source="resume_reconstructed")
            _record_resume_event(
                run_dir,
                {
                    "kind": "reconstructed_online_feedback",
                    "attempt_id": row["attempt_id"],
                    "task_id": task.id,
                    "query_index": query_index,
                },
            )

    event = _event_from_row(config, row, candidate, observation, oracle)
    if event is not None:
        history.append(event)
        should_stop = getattr(method, "should_stop", None)
        if should_stop is not None:
            should_stop(task, tuple(history))
    return candidate, observation, event, reconstructed_feedback


def _append_assessment(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    judge: Any,
    row: dict[str, Any],
    task: Any,
    candidate: Candidate,
    observation: TargetObservation,
) -> dict[str, Any]:
    try:
        assessment = judge.assess(task, candidate, observation)
        if not isinstance(assessment, Assessment):
            raise TypeError("judge.assess 必须返回 Assessment")
        assessment = _apply_outcome_contract(config, observation, assessment)
    except Exception as exc:
        assessment = Assessment(
            success=None,
            bypassed=None,
            details={"judge_error_type": type(exc).__name__, "judge_error": str(exc)},
        )
    assessment_row = {"attempt_id": row["attempt_id"], "task_id": task.id, "assessment": jsonable(assessment)}
    append_jsonl(run_dir / "assessments.jsonl", assessment_row)
    return assessment_row


def run_experiment(
    config_path: str | Path,
    output_root: str | Path = "results",
    *,
    task_ids: list[str] | None = None,
    task_shard_index: int | None = None,
    task_shards: int | None = None,
) -> Path:
    started_monotonic = time.perf_counter()
    root = _root()
    config = load_config(config_path)
    full_tasks = load_tasks(config.dataset_path)
    tasks = _select_task_ids(full_tasks, task_ids)
    tasks = _select_task_shard(tasks, task_shard_index, task_shards)
    if not tasks:
        raise ValueError("任务分片为空")
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    run_dir = _new_run_dir(output_root, config.protocol_id)
    context = RuntimeContext(root=root, run_dir=run_dir)
    manifest = _manifest_for_new_run(
        root=root,
        config=config,
        full_tasks=full_tasks,
        tasks=tasks,
        task_ids=task_ids,
        task_shard_index=task_shard_index,
        task_shards=task_shards,
    )
    write_json_once(run_dir / "manifest.json", manifest)
    write_json_once(run_dir / "config.snapshot.json", json.loads(config.path.read_text(encoding="utf-8")))
    if config.protocol_lock:
        write_json_once(run_dir / "protocol.snapshot.json", config.protocol_lock.payload)
    with (run_dir / "dataset.snapshot.jsonl").open("xb") as handle:
        handle.write(config.dataset_path.read_bytes())
    try:
        components = _build_components(config, context)
    except Exception as exc:
        write_json_once(
            run_dir / "initialization_error.json",
            {"error_type": type(exc).__name__, "error": str(exc), "target_queries": 0},
        )
        write_json_once(run_dir / "completion.json", {"completed": False, "target_queries": 0})
        raise
    method = components.method
    target = components.target
    judge = components.judge
    online_oracle = components.online_oracle
    method_query_limit = components.method_query_limit
    attempts: list[dict[str, Any]] = []
    deferred: list[tuple[dict[str, Any], Any, Any, TargetObservation]] = []
    oracle_calls = 0

    for task in tasks:
        history: list[PolicyEvent] = []
        for query_index in range(1, config.attempts_per_task + 1):
            if method_query_limit is not None and query_index > method_query_limit:
                break
            should_stop = getattr(method, "should_stop", None)
            if should_stop is not None and should_stop(task, tuple(history)):
                break
            generation_seed = _derive_seed(config.experiment_seed, task.id, query_index)
            candidate = method.propose(task, tuple(history), query_index, Random(generation_seed))
            if not isinstance(candidate.prompt, str) or not candidate.prompt.strip():
                raise ValueError(f"方法为 {task.id} 的第 {query_index} 次查询返回空提示词")
            row, observation = _generate_attempt_row(
                config=config,
                run_dir=run_dir,
                target=target,
                task=task,
                candidate=candidate,
                query_index=query_index,
            )
            append_jsonl(run_dir / "attempts.jsonl", row)
            attempts.append(row)
            deferred.append((row, task, candidate, observation))

            if config.feedback_mode is FeedbackMode.TARGET_OBSERVABLE:
                history.append(PolicyEvent(query_index, candidate.prompt, observation.policy_view()))
            elif config.feedback_mode is FeedbackMode.ORACLE_ASSISTED:
                oracle_feedback = online_oracle.feedback(task, candidate, observation)
                oracle_calls += 1
                _append_online_feedback(run_dir, row, oracle_feedback, source="live")
                history.append(
                    PolicyEvent(query_index, candidate.prompt, observation.policy_view(), dict(oracle_feedback))
                )
        complete_task = getattr(method, "complete_task", None)
        if complete_task is not None:
            complete_task(task, tuple(history))

    assessments: list[dict[str, Any]] = []
    for row, task, candidate, observation in deferred:
        assessments.append(
            _append_assessment(
                config=config,
                run_dir=run_dir,
                judge=judge,
                row=row,
                task=task,
                candidate=candidate,
                observation=observation,
            )
        )

    metrics = compute_metrics(
        (task.id for task in tasks),
        attempts,
        assessments,
        config.checkpoints,
        config.attempts_per_task,
        config.denominator_policy,
        (task.id for task in tasks),
    )
    metrics["online_oracle_calls"] = oracle_calls
    metrics["posthoc_judge_calls"] = len(assessments)
    metrics["component_usage"] = {
        "method": _component_usage(method),
        "target": _component_usage(target),
        "online_oracle": _component_usage(online_oracle) if online_oracle else {},
        "posthoc_judge": _component_usage(judge),
    }
    metrics["wall_clock_seconds"] = time.perf_counter() - started_monotonic
    write_json_once(run_dir / "metrics.json", metrics)
    write_json_once(
        run_dir / "completion.json",
        {
            "completed": True,
            "target_queries": len(attempts),
            "planned_target_queries": len(tasks) * config.attempts_per_task,
            "online_oracle_calls": oracle_calls,
            "posthoc_judge_calls": len(assessments),
            "component_usage": metrics["component_usage"],
            "wall_clock_seconds": metrics["wall_clock_seconds"],
        },
    )
    return run_dir


def resume_experiment(
    run_dir: str | Path,
    *,
    allow_config_drift: bool = False,
    config_override_path: str | Path | None = None,
    extend_budget: bool = False,
) -> Path:
    started_monotonic = time.perf_counter()
    root = _root()
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir 不存在：{run_dir}")

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 manifest.json，不能 resume：{run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = Path(str(config_override_path)) if config_override_path is not None else Path(str(manifest.get("config_path", "")))
    if not config_path.is_file():
        raise FileNotFoundError(f"resume config_path 不存在，不能 resume：{config_path}")
    config = load_config(config_path)
    _, tasks, manifest = _validate_resume_manifest(
        run_dir,
        config,
        manifest,
        allow_config_drift=allow_config_drift,
        extend_budget=extend_budget,
    )

    _record_resume_event(
        run_dir,
        {
            "kind": "resume_started",
            "allow_config_drift": allow_config_drift,
            "extend_budget": extend_budget,
            "config_override_path": str(config_path) if config_override_path is not None else None,
            "manifest_config_path": manifest.get("config_path"),
            "manifest_config_sha256": manifest.get("config_sha256"),
            "active_config_sha256": sha256_file(config.path),
            "git_commit": _git(root, "rev-parse", "HEAD"),
            "git_status": _git(root, "status", "--porcelain"),
            "argv": sys.argv,
        },
    )

    context = RuntimeContext(root=root, run_dir=run_dir)
    components = _build_components(config, context)
    method = components.method
    target = components.target
    judge = components.judge
    online_oracle = components.online_oracle
    method_query_limit = components.method_query_limit

    attempts, attempts_by_task = _index_attempts(run_dir, config)
    enrolled_task_ids = {task.id for task in tasks}
    extra_task_ids = sorted(set(attempts_by_task) - enrolled_task_ids)
    if extra_task_ids:
        raise ValueError(f"attempts.jsonl 包含 manifest 任务选择之外的 task_id：{extra_task_ids[:5]}")
    online_feedback = _load_online_feedback(run_dir)
    existing_online_feedback_count = len(online_feedback)
    assessments, assessed_ids = _load_assessments(run_dir)
    preexisting_attempts = len(attempts)
    preexisting_assessments = len(assessments)
    reconstructed_online_feedback = 0
    deferred: list[tuple[dict[str, Any], Any, Candidate, TargetObservation]] = []

    for task in tasks:
        history: list[PolicyEvent] = []
        existing_rows = attempts_by_task.get(task.id, [])
        for row in existing_rows:
            query_index = int(row["query_index"])
            if method_query_limit is not None and query_index > method_query_limit:
                raise ValueError(
                    f"已有 attempt 超过 method.maximum_queries：{task.id}#{query_index}"
                )
            candidate, observation, _, reconstructed = _replay_existing_attempt(
                config=config,
                run_dir=run_dir,
                method=method,
                online_oracle=online_oracle,
                task=task,
                row=row,
                history=history,
                online_feedback=online_feedback,
            )
            reconstructed_online_feedback += int(reconstructed)
            deferred.append((row, task, candidate, observation))

        for query_index in range(len(existing_rows) + 1, config.attempts_per_task + 1):
            if method_query_limit is not None and query_index > method_query_limit:
                break
            should_stop = getattr(method, "should_stop", None)
            if should_stop is not None and should_stop(task, tuple(history)):
                break
            generation_seed = _derive_seed(config.experiment_seed, task.id, query_index)
            candidate = method.propose(task, tuple(history), query_index, Random(generation_seed))
            if not isinstance(candidate.prompt, str) or not candidate.prompt.strip():
                raise ValueError(f"方法为 {task.id} 的第 {query_index} 次查询返回空提示词")
            row, observation = _generate_attempt_row(
                config=config,
                run_dir=run_dir,
                target=target,
                task=task,
                candidate=candidate,
                query_index=query_index,
            )
            append_jsonl(run_dir / "attempts.jsonl", row)
            attempts.append(row)
            deferred.append((row, task, candidate, observation))

            oracle_feedback = None
            if config.feedback_mode is FeedbackMode.ORACLE_ASSISTED:
                if online_oracle is None:
                    raise RuntimeError("oracle_assisted resume 缺少 online_oracle")
                oracle_feedback = dict(online_oracle.feedback(task, candidate, observation))
                online_feedback[row["attempt_id"]] = oracle_feedback
                _append_online_feedback(run_dir, row, oracle_feedback, source="resume_live")
            event = _event_from_row(config, row, candidate, observation, oracle_feedback)
            if event is not None:
                history.append(event)

        complete_task = getattr(method, "complete_task", None)
        if complete_task is not None:
            complete_task(task, tuple(history))

    for row, task, candidate, observation in deferred:
        if row["attempt_id"] in assessed_ids:
            continue
        assessment_row = _append_assessment(
            config=config,
            run_dir=run_dir,
            judge=judge,
            row=row,
            task=task,
            candidate=candidate,
            observation=observation,
        )
        assessments.append(assessment_row)
        assessed_ids.add(row["attempt_id"])

    metrics = compute_metrics(
        (task.id for task in tasks),
        attempts,
        assessments,
        config.checkpoints,
        config.attempts_per_task,
        config.denominator_policy,
        (task.id for task in tasks),
    )
    metrics["online_oracle_calls"] = len(online_feedback) if config.feedback_mode is FeedbackMode.ORACLE_ASSISTED else 0
    metrics["posthoc_judge_calls"] = len(assessments)
    metrics["component_usage"] = {
        "method": _component_usage(method),
        "target": _component_usage(target),
        "online_oracle": _component_usage(online_oracle) if online_oracle else {},
        "posthoc_judge": _component_usage(judge),
    }
    metrics["resume"] = {
        "preexisting_attempts": preexisting_attempts,
        "preexisting_assessments": preexisting_assessments,
        "existing_online_feedback_records": existing_online_feedback_count,
        "reconstructed_online_feedback": reconstructed_online_feedback,
        "new_attempts": len(attempts) - preexisting_attempts,
        "new_assessments": len(assessments) - preexisting_assessments,
    }
    metrics["wall_clock_seconds"] = time.perf_counter() - started_monotonic
    write_json_once(run_dir / "metrics.json", metrics)
    write_json_once(
        run_dir / "completion.json",
        {
            "completed": True,
            "resumed": True,
            "target_queries": len(attempts),
            "planned_target_queries": len(tasks) * config.attempts_per_task,
            "online_oracle_calls": metrics["online_oracle_calls"],
            "posthoc_judge_calls": len(assessments),
            "component_usage": metrics["component_usage"],
            "resume": metrics["resume"],
            "wall_clock_seconds": metrics["wall_clock_seconds"],
        },
    )
    _record_resume_event(
        run_dir,
        {
            "kind": "resume_completed",
            "target_queries": len(attempts),
            "new_attempts": len(attempts) - preexisting_attempts,
            "reconstructed_online_feedback": reconstructed_online_feedback,
        },
    )
    return run_dir
