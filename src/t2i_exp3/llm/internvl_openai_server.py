from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a local InternVL HF model through OpenAI-compatible text-only chat endpoints."
    )
    parser.add_argument("--model-path", required=True, help="Local InternVL HF snapshot directory.")
    parser.add_argument("--served-model-name", default=None, help="Model id exposed by /v1/models.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cpu, or auto.")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-remote-files", action="store_true")
    return parser


def _normalise_messages(messages_raw: list[Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages_raw):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] 必须为对象")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{index}].role 必须为字符串")
        if isinstance(content, str):
            normalised_content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            normalised_content = content
        else:
            raise ValueError(f"messages[{index}].content 必须为字符串或 OpenAI content parts")
        messages.append({"role": role, "content": normalised_content})
    return messages


def _seed_generation(torch_module: Any, *, model_device: Any, seed: int | None) -> None:
    if seed is None:
        return
    torch_module.manual_seed(seed)
    if getattr(model_device, "type", None) == "cuda":
        torch_module.cuda.manual_seed_all(seed)


def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved
    return inputs


def create_app(
    *,
    model_path: str | Path,
    served_model_name: str | None = None,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_new_tokens_default: int = 512,
    trust_remote_code: bool = False,
    local_files_only: bool = True,
):
    try:
        import torch
        from fastapi import Body, FastAPI, HTTPException
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "启动本地 InternVL OpenAI-compatible 服务需要安装 torch/transformers/fastapi/uvicorn"
        ) from exc

    model_dir = Path(model_path).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if max_new_tokens_default < 1:
        raise ValueError("max_new_tokens_default 必须为正整数")
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_by_name[dtype]
    exposed_model = served_model_name or model_dir.name
    config = AutoConfig.from_pretrained(
        str(model_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    architectures = tuple(getattr(config, "architectures", None) or ())
    hf_internvl = "InternVLForConditionalGeneration" in architectures

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    processor = None
    tokenizer = None
    backend = "transformers.AutoModelForCausalLM"
    if hf_internvl:
        processor = AutoProcessor.from_pretrained(
            str(model_dir),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        backend = "transformers.AutoModelForImageTextToText"
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    if device == "auto":
        model_kwargs["device_map"] = "auto"
        model_cls = AutoModelForImageTextToText if hf_internvl else AutoModelForCausalLM
        model = model_cls.from_pretrained(str(model_dir), **model_kwargs)
    else:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"请求 {device}，但 CUDA 不可用")
        model_cls = AutoModelForImageTextToText if hf_internvl else AutoModelForCausalLM
        model = model_cls.from_pretrained(str(model_dir), **model_kwargs).to(device)
    model.eval()

    app = FastAPI(title="Local InternVL OpenAI-compatible service")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": exposed_model,
            "model_path": str(model_dir),
            "backend": backend,
            "device": device,
            "dtype": dtype,
            "mode": "text_only",
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": exposed_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="stream=true 暂不支持")
        messages_raw = payload.get("messages")
        if not isinstance(messages_raw, list) or not messages_raw:
            raise HTTPException(status_code=400, detail="messages 不能为空")
        try:
            messages = _normalise_messages(messages_raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        max_tokens_raw = payload.get("max_tokens")
        max_new_tokens = int(max_tokens_raw) if max_tokens_raw is not None else max_new_tokens_default
        if max_new_tokens < 1:
            raise HTTPException(status_code=400, detail="max_tokens 必须为正整数")
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        seed_raw = payload.get("seed")
        seed = int(seed_raw) if seed_raw is not None else None

        model_device = next(model.parameters()).device
        question_parts: list[str] = []
        for message in messages:
            content = message["content"]
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            role = str(message["role"]).strip()
            text = "\n".join(text_parts).strip()
            if text:
                question_parts.append(f"{role}: {text}")
        question = "\n\n".join(question_parts).strip()
        if not question:
            raise HTTPException(status_code=400, detail="messages 缺少可用文本内容")
        do_sample = temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        else:
            generation_kwargs["top_k"] = None
        if model_device.type == "cuda":
            torch.cuda.synchronize(model_device)
        started = time.perf_counter()
        _seed_generation(torch, model_device=model_device, seed=seed if do_sample else None)
        with torch.inference_mode():
            if hf_internvl:
                assert processor is not None
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = _move_inputs_to_device(inputs, model_device)
                outputs = model.generate(**inputs, **generation_kwargs)
                input_length = int(inputs["input_ids"].shape[-1])
                generated = outputs[0][input_length:]
                text = processor.decode(generated, skip_special_tokens=True)
            else:
                assert tokenizer is not None
                text = model.chat(
                    tokenizer,
                    pixel_values=None,
                    question=question,
                    generation_config=generation_kwargs,
                    history=None,
                    return_history=False,
                )
        if model_device.type == "cuda":
            torch.cuda.synchronize(model_device)
        token_counter = tokenizer if tokenizer is not None else processor.tokenizer
        prompt_tokens = len(token_counter(question, add_special_tokens=False).input_ids)
        completion_tokens = len(token_counter(str(text), add_special_tokens=False).input_ids)
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": exposed_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "system_fingerprint": {
                "backend": backend,
                "model_path": str(model_dir),
                "device": str(model_device),
                "dtype": dtype,
                "mode": "text_only",
                "latency_seconds": round(time.perf_counter() - started, 6),
            },
        }

    return app


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("启动服务需要安装 uvicorn") from exc
    app = create_app(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens_default=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        local_files_only=not args.allow_remote_files,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
