#!/usr/bin/env python3
"""Run the manuscript configurations or a short smoke check."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-root", type=Path, default=Path("reruns"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/experiments.json").read_text(encoding="utf-8"))
    jobs = {job["name"]: job for group in config.values() for job in group}
    if args.list or not args.stages:
        for group, entries in config.items():
            print(f"{group}:")
            for job in entries:
                print(f"  {job['name']:<24} {job['description']}")
        return
    selected = []
    for stage in args.stages:
        if stage == "all":
            entries = config["paper"]
        elif stage == "smoke":
            entries = config["smoke"]
        elif stage in jobs:
            entries = [jobs[stage]]
        else:
            parser.error(f"Unknown stage: {stage}; use --list")
        for job in entries:
            if job not in selected:
                selected.append(job)
    out_root = (ROOT / args.out_root).resolve()
    archive = ROOT / "runs"
    if out_root == archive or archive in out_root.parents:
        parser.error("The bundled runs/ directory is reserved for archived results; choose another --out-root")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
    for job in selected:
        out = out_root / job["outdir"]
        command = [sys.executable, str(ROOT / job["script"]), *job["arguments"]]
        command += ["--device", args.device]
        if job["name"] == "mc":
            command += ["--runs-root", str(out_root)]
        command += ["--outdir", str(out)]
        display = ["python", job["script"], *command[2:]]
        print(f"\n{job['name']}: {shlex.join(display)}", flush=True)
        if args.dry_run:
            continue
        if out.exists() and any(out.iterdir()) and not args.overwrite:
            parser.error(f"Output already exists: {out}; choose a fresh directory or use --overwrite")
        out.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": job["name"], "script": job["script"],
            "arguments": job["arguments"], "device": args.device,
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "numpy", "matplotlib", "psutil")},
            "status": "running",
        }
        config_path = out / "run_config.json"
        config_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        start = time.perf_counter()
        with (out / "execution.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            returncode = process.wait()
        record.update(status="completed" if returncode == 0 else "failed",
                      returncode=returncode, elapsed_seconds=time.perf_counter() - start)
        config_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if returncode:
            raise SystemExit(returncode)


if __name__ == "__main__":
    main()
