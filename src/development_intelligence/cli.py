"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .ollama_client import OllamaClient
from .pipeline import AnalysisRunner, load_or_prepare, prepare_document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Timor-Leste development intelligence pipeline")
    parser.add_argument("command", choices=("prepare", "check-models", "smoke", "run", "compare", "all"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--limit-chunks", type=int, default=None, help="Smoke-test only; omit for final analysis")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = PipelineConfig(project_root=args.project_root.resolve())
    config.create_directories()
    if args.command == "prepare":
        result = prepare_document(config)
    elif args.command == "check-models":
        client = OllamaClient(config.ollama_url, config.cache_dir)
        result = {"available_models": client.available_models()}
    elif args.command == "run":
        load_or_prepare(config)
        result = AnalysisRunner(config).run(limit_chunks=args.limit_chunks)
    elif args.command == "smoke":
        load_or_prepare(config)
        result = AnalysisRunner(config).smoke_test()
    elif args.command == "compare":
        load_or_prepare(config)
        result = AnalysisRunner(config).run_comparison()
    else:
        prepare_document(config)
        result = AnalysisRunner(config).run(limit_chunks=args.limit_chunks)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
