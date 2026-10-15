"""Explicit, checksum-verified downloads; never copy personal runtime assets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen


def download(url: str, path: Path, sha256: str) -> None:
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(url, timeout=120) as response:
            content = response.read()
    actual = hashlib.sha256(content).hexdigest()
    if actual != sha256:
        raise ValueError(f"Checksum mismatch for {path.name}: {actual}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)


def prepare_i2p(csv_path: Path, selection: dict, output: Path) -> dict:
    source = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8-sig"))))
    wanted = set(selection["task_ids"])
    selected = []
    for row in source:
        id = f"i2p-{int(row['case_number']):04d}"
        if id not in wanted:
            continue
        categories = [c.strip().lower() for c in row["categories"].split(",") if c.strip()]
        prompt = row["prompt"].strip()
        if not prompt or not categories:
            raise ValueError(f"Invalid selected record: {id}")
        selected.append({"id": id, "intent": prompt, "category": categories[0], "metadata": {
            "case_number": row["case_number"].strip(), "categories": categories,
            "evaluation_seed": int(row["evaluation_seed"]),
            "evaluation_guidance": float(row["evaluation_guidance"]),
            "width": int(row["sd_image_width"]), "height": int(row["sd_image_height"]),
            "source_model": row["sd_model"].strip()}})
    order = {id: i for i, id in enumerate(selection["task_ids"])}
    selected.sort(key=lambda r: order[r["id"]])
    if len(selected) != len(wanted) or len({r["id"] for r in selected}) != len(wanted):
        raise ValueError("Missing or duplicate selected I2P case numbers")
    content = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n' for r in selected).encode()
    digest = hashlib.sha256(content).hexdigest()
    if digest != selection["prepared_sha256"]:
        raise ValueError(f"Prepared dataset checksum mismatch: {digest}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != content:
        raise FileExistsError(f"Refusing to overwrite different dataset: {output}")
    output.write_bytes(content)
    return {"records": len(selected), "sha256": digest, "output": str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("resource", choices=["i2p", "evaluators"])
    p.add_argument("--root", type=Path, default=Path.cwd())
    a = p.parse_args()
    root = a.root.resolve()
    manifest = json.loads((root / "data/resources.json").read_text())
    entries = manifest["i2p_sources"] if a.resource == "i2p" else manifest["evaluator_files"]
    for entry in entries:
        download(entry["url"], root / entry["path"], entry["sha256"])
    if a.resource == "i2p":
        selection = json.loads((root / "data/i2p200_selection.json").read_text())
        print(json.dumps(prepare_i2p(root / entries[0]["path"], selection, root / "data/prepared/i2p200.jsonl")))
    else:
        print(json.dumps({"verified_files": len(entries)}))
