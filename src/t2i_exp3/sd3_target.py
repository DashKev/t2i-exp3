from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .qwen_image_target import _InputGuard, _OutputChecker, _build_input_guard, _build_output_checker
from .schema import Candidate, TargetObservation, TargetStatus, Task


class StableDiffusion3GuardedTarget:
    """Stable Diffusion 3/3.5 Diffusers target with explicit external guards.

    SD3.5-Medium does not expose an SD1.4-style native safety checker in the
    local Diffusers deployment.  This class therefore models the deployed
    system boundary explicitly as:

    prompt input guard -> SD3 image generator -> output image/text-image guard.

    L2/L3 remain post-hoc benchmark metrics and are intentionally computed by
    the configured judge, not by this target.
    """

    def __init__(
        self,
        parameters: dict[str, Any],
        context: Any,
        *,
        pipe: Any | None = None,
        torch_module: Any | None = None,
        input_guard: _InputGuard | None = None,
        output_checker: _OutputChecker | None = None,
    ) -> None:
        if torch_module is None:
            import torch

            torch_module = torch
        self.torch = torch_module
        self.target_id = str(parameters.get("target_id", "sd3_guarded_diffusers"))
        self.device = str(parameters.get("device", "cuda:0"))
        self.model = _resolve_model_ref(
            parameters.get("model_path_or_id", "stabilityai/stable-diffusion-3.5-medium"),
            context.root,
        )
        self.local_files_only = bool(parameters.get("local_files_only", True))
        self.cpu_offload = bool(parameters.get("cpu_offload", True))
        self.cpu_offload_mode = str(parameters.get("cpu_offload_mode", "model"))
        if self.cpu_offload_mode not in {"model", "sequential"}:
            raise ValueError("cpu_offload_mode 必须为 model 或 sequential")
        raw_device_map = parameters.get("device_map")
        self.device_map = None if raw_device_map in (None, "", "none", "None") else raw_device_map
        self.max_memory = _normalize_max_memory(parameters.get("max_memory"))
        if self.device_map is not None and self.cpu_offload:
            raise ValueError("device_map 与 cpu_offload 不能同时启用")
        self.generator_device = str(
            parameters.get("generator_device", "cpu" if self.cpu_offload or self.device_map is not None else self.device)
        )
        self.steps = int(parameters.get("num_inference_steps", 28))
        self.guidance = float(parameters.get("guidance_scale", 4.5))
        self.width = int(parameters.get("width", 1024))
        self.height = int(parameters.get("height", 1024))
        self.max_sequence_length = int(parameters.get("max_sequence_length", 256))
        self.negative_prompt = parameters.get("negative_prompt")
        self.archive_blocked_artifact = bool(parameters.get("archive_blocked_artifact", False))
        self._input_guard = input_guard or _build_input_guard(
            dict(parameters.get("input_guard", {"mode": "disabled"})),
            context,
        )
        self._output_checker = output_checker or _build_output_checker(
            dict(parameters.get("output_checker", {"mode": "disabled"})),
            context,
        )

        self._calls = 0
        self._returned = 0
        self._refused = 0
        self._blocked = 0
        if pipe is None:
            from diffusers import StableDiffusion3Pipeline

            dtype = _torch_dtype(self.torch, str(parameters.get("torch_dtype", "float16")))
            kwargs: dict[str, Any] = {
                "local_files_only": self.local_files_only,
            }
            if dtype != "auto":
                kwargs["torch_dtype"] = dtype
            if self.device_map is not None:
                kwargs["device_map"] = self.device_map
                if self.max_memory:
                    kwargs["max_memory"] = self.max_memory
            pipe = StableDiffusion3Pipeline.from_pretrained(self.model, **kwargs)
            if self.device_map is not None:
                pass
            elif self.cpu_offload:
                if self.cpu_offload_mode == "sequential":
                    pipe.enable_sequential_cpu_offload(device=self.device)
                else:
                    pipe.enable_model_cpu_offload(device=self.device)
            else:
                pipe = pipe.to(self.device)
        self.pipe = pipe
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

    def generate(
        self,
        task: Task,
        candidate: Candidate,
        generation_seed: int,
        artifact_dir: Path,
    ) -> TargetObservation:
        self._calls += 1
        guard_decision = self._input_guard.check(candidate.prompt)
        private: dict[str, Any] = {
            "target_id": self.target_id,
            "target": "stable_diffusion_3_diffusers",
            "target_family": "open_composed_guarded_t2i",
            "generator": "Stable-Diffusion-3.5-Medium",
            "model": self.model,
            "local_files_only": self.local_files_only,
            "cpu_offload": self.cpu_offload,
            "cpu_offload_mode": self.cpu_offload_mode,
            "device_map": self.device_map,
            "max_memory": self.max_memory,
            "generator_device": self.generator_device,
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance,
            "width": self.width,
            "height": self.height,
            "max_sequence_length": self.max_sequence_length,
            "native_safety_filter": False,
            "target_side_safety_filter": True,
            "guardrail_boundary": "input_prompt_moderation+output_image_and_text_image_moderation",
            "input_guard": guard_decision.jsonable(),
        }
        if guard_decision.blocked:
            self._refused += 1
            return TargetObservation(
                status=TargetStatus.REFUSED,
                public_feedback={
                    "result": "refused",
                    "reason": "input_safety_guard",
                    "safety_label": guard_decision.label,
                },
                private_metadata=private,
            )

        generator = self.torch.Generator(device=self.generator_device).manual_seed(generation_seed)
        kwargs: dict[str, Any] = {
            "prompt": candidate.prompt,
            "generator": generator,
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance,
            "width": self.width,
            "height": self.height,
            "max_sequence_length": self.max_sequence_length,
        }
        if self.negative_prompt is not None:
            kwargs["negative_prompt"] = str(self.negative_prompt)
        result = self.pipe(**kwargs)
        image = result.images[0]
        output_decision = self._output_checker.check(image, prompt=candidate.prompt, task=task)
        private["output_checker"] = output_decision.jsonable()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if output_decision.blocked:
            self._blocked += 1
            paths: tuple[str, ...] = ()
            if self.archive_blocked_artifact:
                filename = (
                    hashlib.sha256(f"{task.id}:{generation_seed}:sd3:blocked".encode()).hexdigest()[:24]
                    + ".png"
                )
                output = artifact_dir / filename
                image.save(output)
                paths = (str(output.relative_to(artifact_dir.parent)),)
            return TargetObservation(
                status=TargetStatus.BLOCKED,
                artifact_paths=paths,
                public_feedback={
                    "result": "blocked",
                    "reason": "output_safety_checker",
                    "safety_label": output_decision.label,
                },
                private_metadata=private,
            )

        filename = hashlib.sha256(f"{task.id}:{generation_seed}:sd3".encode()).hexdigest()[:24] + ".png"
        output = artifact_dir / filename
        image.save(output)
        self._returned += 1
        return TargetObservation(
            status=TargetStatus.RETURNED,
            artifact_paths=(str(output.relative_to(artifact_dir.parent)),),
            public_feedback={"result": "artifact_returned"},
            private_metadata=private,
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "calls": self._calls,
            "returned": self._returned,
            "refused": self._refused,
            "blocked": self._blocked,
            "model": self.model,
            "device": self.device,
            "max_sequence_length": self.max_sequence_length,
            "cpu_offload": self.cpu_offload,
            "cpu_offload_mode": self.cpu_offload_mode,
            "device_map": self.device_map,
            "max_memory": self.max_memory,
            "generator_device": self.generator_device,
            "native_safety_filter": False,
            "target_side_safety_filter": True,
            "input_guard": self._input_guard.usage_snapshot(),
            "output_checker": self._output_checker.usage_snapshot(),
        }


def _resolve_model_ref(raw: Any, root: Path) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("model_path_or_id 必须为非空字符串")
    value = raw.strip()
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return str(path.resolve())
    relative = (root / path).resolve()
    if relative.exists():
        return str(relative)
    return value


def _torch_dtype(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError("torch_dtype 只能为 auto/bfloat16/float16/float32")


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


def build_target(parameters: dict[str, Any], context: Any) -> StableDiffusion3GuardedTarget:
    return StableDiffusion3GuardedTarget(dict(parameters), context)
