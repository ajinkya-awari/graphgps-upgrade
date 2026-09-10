from __future__ import annotations

import argparse

from .config import load_config
from .preflight import offline_preflight
from .static_validation import run_static_validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project 12 gated benchmark CLI.")
    parser.add_argument("--config", default="configs/fixture.toml")
    parser.add_argument("--mode", choices=("preflight", "fixture", "static", "kaggle-smoke", "ablation"), default="preflight")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = offline_preflight(config)
    if not result.ok:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 2
    if args.mode in {"kaggle-smoke", "ablation"}:
        print("ERROR: Kaggle/data/GPU execution requires explicit approval outside local CLI automation")
        return 2
    if args.mode == "static":
        results = run_static_validation(".")
        for static_result in results:
            print(f"{static_result.name}={'ok' if static_result.ok else 'fail'}")
            for finding in static_result.findings:
                print(f"  {finding}")
        return 0 if all(static_result.ok for static_result in results) else 2
    print(f"offline_preflight_ok=true config_hash={result.config_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
