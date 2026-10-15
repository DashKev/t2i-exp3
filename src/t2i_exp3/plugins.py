from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import PluginSpec


@dataclass(frozen=True)
class RuntimeContext:
    root: Path
    run_dir: Path


def _resolve_symbol(reference: str, root: Path) -> Callable[..., Any]:
    if ":" not in reference:
        raise ValueError(f"插件引用缺少 ':'：{reference}")
    module_ref, symbol_name = reference.rsplit(":", 1)
    if module_ref.endswith(".py") or "/" in module_ref:
        module_path = (root / module_ref).resolve()
        if not module_path.is_relative_to(root.resolve()):
            raise ValueError("插件路径不能越出仓库")
        if not module_path.is_file():
            raise ValueError(f"插件文件不存在：{module_path}")
        module_name = f"t2i_bench_plugin_{module_path.stem}_{abs(hash(module_path))}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"无法加载插件：{module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    else:
        module = importlib.import_module(module_ref)
    symbol = getattr(module, symbol_name, None)
    if not callable(symbol):
        raise ValueError(f"插件符号不存在或不可调用：{reference}")
    return symbol


def build_plugin(spec: PluginSpec, context: RuntimeContext) -> Any:
    return _resolve_symbol(spec.factory, context.root)(dict(spec.parameters), context)
