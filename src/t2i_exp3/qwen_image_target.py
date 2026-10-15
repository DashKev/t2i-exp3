from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .llm.openai_compatible import OpenAICompatibleClient
from .schema import Candidate, TargetObservation, TargetStatus, Task


_QWEN3GUARD_CATEGORIES = (
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
    "None",
)
_SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controvers\w*)", re.IGNORECASE)
_CATEGORY_RE = re.compile("|".join(re.escape(item) for item in _QWEN3GUARD_CATEGORIES))


def _prefer_local_bin_when_no_safetensors(model_path_or_id: str) -> dict[str, bool]:
    """Return loader kwargs for local model folders that only contain PyTorch bins.

    Some of our cached safety-checker assets are stored as ``pytorch_model.bin``
    without a matching ``model.safetensors``. Newer Transformers/Diffusers
    versions may prefer safetensors and fail before falling back, so make the
    format choice explicit for local folders. Remote model ids are left
    untouched.
    """

    path = Path(model_path_or_id)
    if path.is_dir() and not (path / "model.safetensors").exists() and (path / "pytorch_model.bin").exists():
        return {"use_safetensors": False}
    return {}


@dataclass(frozen=True)
class InputGuardDecision:
    label: str
    categories: tuple[str, ...]
    blocked: bool
    raw_output: str
    guard: str
    model: str | None = None

    def jsonable(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "categories": list(self.categories),
            "blocked": self.blocked,
            "raw_output": self.raw_output,
            "guard": self.guard,
            "model": self.model,
        }


@dataclass(frozen=True)
class OutputCheckerDecision:
    blocked: bool
    label: str
    checker: str
    model: str | None = None
    details: dict[str, Any] | None = None

    def jsonable(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "label": self.label,
            "checker": self.checker,
            "model": self.model,
            "details": dict(self.details or {}),
        }


class _InputGuard(Protocol):
    def check(self, prompt: str) -> InputGuardDecision: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class _OutputChecker(Protocol):
    def check(
        self,
        image: Any,
        *,
        prompt: str | None = None,
        task: Task | None = None,
    ) -> OutputCheckerDecision: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class _ImageGenerator(Protocol):
    def generate_image(self, prompt: str, generation_seed: int) -> tuple[Any, dict[str, Any]]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def parse_qwen3guard_output(
    content: str,
    *,
    block_labels: set[str] | None = None,
    guard: str = "qwen3guard",
    model: str | None = None,
) -> InputGuardDecision:
    """Parse Qwen3Guard-Gen's structured text output.

    The official examples emit lines like:

    ``Safety: Unsafe``
    ``Categories: Violent``
    """

    label_match = _SAFETY_RE.search(content)
    if label_match is None:
        raise ValueError(f"Qwen3Guard 输出缺少 Safety 标签：{content[:200]!r}")
    label = label_match.group(1).title()
    if label.lower().startswith("controvers") and label.lower() != "controversial":
        # Qwen3Guard occasionally emits misspellings such as "Controversal" or
        # "Controversing". Treat them as the documented "Controversial" label
        # instead of failing an otherwise valid guard call; raw_output remains
        # unchanged for auditability.
        label = "Controversial"
    categories = tuple(dict.fromkeys(_CATEGORY_RE.findall(content))) or ("None",)
    normalized_block_labels = {item.lower() for item in (block_labels or {"unsafe"})}
    return InputGuardDecision(
        label=label,
        categories=categories,
        blocked=label.lower() in normalized_block_labels,
        raw_output=content.strip(),
        guard=guard,
        model=model,
    )


class DisabledInputGuard:
    def __init__(self) -> None:
        self._calls = 0

    def check(self, prompt: str) -> InputGuardDecision:
        self._calls += 1
        return InputGuardDecision(
            label="Safe",
            categories=("None",),
            blocked=False,
            raw_output="Safety: Safe\nCategories: None",
            guard="disabled",
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return {"mode": "disabled", "calls": self._calls}


class DisabledOutputChecker:
    def __init__(self) -> None:
        self._calls = 0

    def check(
        self,
        image: Any,
        *,
        prompt: str | None = None,
        task: Task | None = None,
    ) -> OutputCheckerDecision:
        self._calls += 1
        return OutputCheckerDecision(
            blocked=False,
            label="safe",
            checker="disabled",
            details={"family": "disabled"},
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return {"mode": "disabled", "calls": self._calls}


class StableDiffusionSafetyOutputChecker:
    """Image-side output guard based on Diffusers' Stable Diffusion safety checker.

    This is intentionally separate from the unified post-hoc judge.  It models a
    deploy-time output image moderation step and records that the checker is an
    external guard, not native Qwen-Image behavior.
    """

    def __init__(self, parameters: dict[str, Any], context: Any) -> None:
        import numpy as np
        import torch
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor

        self._np = np
        self._torch = torch
        self._device = str(parameters.get("device", "cuda:0"))
        self._model_path_or_id = _resolve_path_or_id(
            context.root,
            str(parameters.get("model_path_or_id", "models/sd14-p4d/safety_checker")),
        )
        self._feature_extractor_path_or_id = _resolve_path_or_id(
            context.root,
            str(parameters.get("feature_extractor_path_or_id", "models/sd14-p4d/feature_extractor")),
        )
        self._local_files_only = bool(parameters.get("local_files_only", True))
        torch_dtype = _torch_dtype(torch, str(parameters.get("torch_dtype", "float16")))
        kwargs: dict[str, Any] = {"local_files_only": self._local_files_only}
        if torch_dtype != "auto":
            kwargs["torch_dtype"] = torch_dtype
        kwargs.update(_prefer_local_bin_when_no_safetensors(self._model_path_or_id))
        self._checker = StableDiffusionSafetyChecker.from_pretrained(
            self._model_path_or_id,
            **kwargs,
        ).to(self._device)
        self._feature_extractor = CLIPImageProcessor.from_pretrained(
            self._feature_extractor_path_or_id,
            local_files_only=self._local_files_only,
        )
        self._calls = 0

    def check(
        self,
        image: Any,
        *,
        prompt: str | None = None,
        task: Task | None = None,
    ) -> OutputCheckerDecision:
        self._calls += 1
        rgb = image.convert("RGB")
        images = self._np.expand_dims(self._np.asarray(rgb).astype(self._np.float32) / 255.0, axis=0)
        clip_input = self._feature_extractor([rgb], return_tensors="pt").pixel_values.to(self._device)
        with self._torch.inference_mode():
            _, flags = self._checker(images=images, clip_input=clip_input)
        flag = bool(flags and flags[0])
        return OutputCheckerDecision(
            blocked=flag,
            label="unsafe" if flag else "safe",
            checker="stable_diffusion_safety_checker",
            model=self._model_path_or_id,
            details={
                "family": "image",
                "nsfw_content_detected": flag,
                "feature_extractor_path_or_id": self._feature_extractor_path_or_id,
                "external_safety_filter": True,
            },
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "mode": "stable_diffusion_safety_checker",
            "model_path_or_id": self._model_path_or_id,
            "feature_extractor_path_or_id": self._feature_extractor_path_or_id,
            "device": self._device,
            "calls": self._calls,
        }




class CompositeOutputChecker:
    """Run multiple target-side output checker families in order."""

    def __init__(self, checkers: list[_OutputChecker]) -> None:
        if not checkers:
            raise ValueError("composite output checker 至少需要一个子 checker")
        self._checkers = checkers
        self._calls = 0

    def check(
        self,
        image: Any,
        *,
        prompt: str | None = None,
        task: Task | None = None,
    ) -> OutputCheckerDecision:
        self._calls += 1
        sub_checks: list[dict[str, Any]] = []
        blocked_by: dict[str, Any] | None = None
        for checker in self._checkers:
            decision = checker.check(image, prompt=prompt, task=task)
            item = decision.jsonable()
            sub_checks.append(item)
            if decision.blocked and blocked_by is None:
                blocked_by = item
        blocked = blocked_by is not None
        return OutputCheckerDecision(
            blocked=blocked,
            label=str(blocked_by.get("label", "unsafe")) if blocked_by else "safe",
            checker="composite_output_checker",
            details={
                "family": "composite",
                "sub_checks": sub_checks,
                "blocked_by": blocked_by,
            },
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "mode": "composite",
            "calls": self._calls,
            "checkers": [checker.usage_snapshot() for checker in self._checkers],
        }


class OpenAICompatibleQwen3Guard:
    def __init__(self, parameters: dict[str, Any]) -> None:
        self._model = str(parameters["model"])
        self._client = OpenAICompatibleClient(
            base_url=str(parameters["base_url"]).rstrip("/"),
            api_key=str(parameters.get("api_key", "EMPTY")),
            model=self._model,
            timeout_seconds=float(parameters.get("timeout_seconds", 300.0)),
        )
        self._max_tokens = int(parameters.get("max_tokens", 128))
        self._block_labels = {str(item).lower() for item in parameters.get("block_labels", ["Unsafe"])}
        self._calls = 0

    def check(self, prompt: str) -> InputGuardDecision:
        self._calls += 1
        response = self._client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        return parse_qwen3guard_output(
            response.text,
            block_labels=self._block_labels,
            guard="qwen3guard_openai_compatible",
            model=response.model or self._model,
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return {"mode": "qwen3guard_openai_compatible", "model": self._model, "calls": self._calls}


class TransformersQwen3Guard:
    def __init__(self, parameters: dict[str, Any], context: Any) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_path_or_id = _resolve_path_or_id(context.root, str(parameters["model_path_or_id"]))
        self._device = str(parameters.get("device", "cuda:0"))
        self._device_map = parameters.get("device_map")
        self._max_new_tokens = int(parameters.get("max_new_tokens", 128))
        self._block_labels = {str(item).lower() for item in parameters.get("block_labels", ["Unsafe"])}
        torch_dtype = _torch_dtype(torch, str(parameters.get("torch_dtype", "auto")))
        tokenizer_kwargs: dict[str, Any] = {
            "local_files_only": bool(parameters.get("local_files_only", True)),
        }
        kwargs: dict[str, Any] = dict(tokenizer_kwargs)
        if torch_dtype != "auto":
            kwargs["dtype"] = torch_dtype
        if self._device_map:
            kwargs["device_map"] = self._device_map
            if "max_memory" in parameters:
                max_memory = parameters["max_memory"]
                if isinstance(max_memory, dict):
                    kwargs["max_memory"] = {
                        int(key) if isinstance(key, str) and key.isdigit() else key: value
                        for key, value in max_memory.items()
                    }
                else:
                    kwargs["max_memory"] = max_memory
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path_or_id, **tokenizer_kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_path_or_id, **kwargs)
        if self._device_map:
            self._input_device = next(self._model.parameters()).device
        else:
            self._model = self._model.to(self._device)
            self._input_device = self._model.device
        self._model.eval()
        self._calls = 0

    def check(self, prompt: str) -> InputGuardDecision:
        self._calls += 1
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._input_device)
        with self._torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        output_ids = outputs[0][len(inputs["input_ids"][0]) :].tolist()
        content = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return parse_qwen3guard_output(
            content,
            block_labels=self._block_labels,
            guard="qwen3guard_transformers",
            model=self._model_path_or_id,
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "mode": "qwen3guard_transformers",
            "model_path_or_id": self._model_path_or_id,
            "device": self._device,
            "device_map": self._device_map,
            "calls": self._calls,
        }






def _build_input_guard(parameters: dict[str, Any], context: Any) -> _InputGuard:
    mode = str(parameters.get("mode", "disabled"))
    if mode == "disabled":
        return DisabledInputGuard()
    if mode == "http_remote":
        from .remote_eval import HttpRemoteInputGuard

        return HttpRemoteInputGuard(parameters, context)
    if mode == "openai_compatible":
        return OpenAICompatibleQwen3Guard(parameters)
    if mode == "qwen3guard_transformers":
        return TransformersQwen3Guard(parameters, context)
    raise ValueError("input_guard.mode 必须为 disabled/http_remote/openai_compatible/qwen3guard_transformers")


def _build_output_checker(parameters: dict[str, Any], context: Any) -> _OutputChecker:
    mode = str(parameters.get("mode", "disabled"))
    if mode == "disabled":
        return DisabledOutputChecker()
    if mode == "http_remote":
        from .remote_eval import HttpRemoteOutputChecker

        return HttpRemoteOutputChecker(parameters, context)
    if mode == "stable_diffusion_safety_checker":
        return StableDiffusionSafetyOutputChecker(parameters, context)
    if mode == "composite":
        raw_checkers = parameters.get("checkers")
        if not isinstance(raw_checkers, list) or not raw_checkers:
            raise ValueError("output_checker.mode=composite 时必须提供非空 checkers 数组")
        return CompositeOutputChecker(
            [_build_output_checker(dict(item), context) for item in raw_checkers]
        )
    raise ValueError(
        "output_checker.mode 必须为 disabled/http_remote/stable_diffusion_safety_checker/"
        "composite"
    )


def _resolve_path_or_id(root: Path, raw: str) -> str:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    local = (root / path).resolve()
    if local.exists():
        return str(local)
    return raw


def _torch_dtype(torch: Any, raw: str) -> Any:
    if raw == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[raw]
    except KeyError as exc:
        raise ValueError(f"不支持的 torch_dtype={raw!r}") from exc


def _normalize_max_memory(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    normalized: dict[Any, Any] = {}
    for key, value in raw.items():
        if isinstance(key, str) and key.isdigit():
            normalized[int(key)] = value
        else:
            normalized[key] = value
    return normalized


