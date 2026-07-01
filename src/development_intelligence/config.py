"""Central configuration for reproducible pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


THEMES = (
    "education",
    "health",
    "inequality",
    "economy",
    "gender",
    "climate",
    "employment",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration values kept in one place for transparent experiments."""

    project_root: Path = field(default_factory=lambda: Path.cwd())
    pdf_filename: str = "timor-Leste.pdf"
    extractor_model: str = "qwen2.5:3b"
    evaluator_model: str = "llama3.2"
    comparison_models: tuple[str, ...] = ("qwen2.5:3b", "llama3.2", "phi3:mini")
    ollama_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    seed: int = 42
    chunk_size: int = 18_000
    chunk_overlap: int = 1_500
    retrieval_top_k: int = 6

    @property
    def pdf_path(self) -> Path:
        return self.project_root / self.pdf_filename

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def processed_dir(self) -> Path:
        return self.output_dir / "processed"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    def create_directories(self) -> None:
        for directory in (
            self.processed_dir,
            self.results_dir,
            self.cache_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
