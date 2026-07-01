# Timor-Leste Local-LLM Development Intelligence

An evidence grounded data analysis pipeline and interactive dashboard for the
**Timor-Leste National Human Development Report 2018**. The project combines
deterministic PDF and table extraction with local language models to identify
development themes, indicators, trends, strengths, and challenges while
retaining source-page traceability.

## Features

- Extracts text, sections, tables, and page-aware evidence chunks from the PDF.
- Classifies evidence across education, health, inequality, economy, gender,
  climate, and employment themes.
- Produces structured indicators, time series, chapter summaries, and findings.
- Separates verified development strengths from current challenges.
- Evaluates generated outputs for factual alignment, completeness, consistency,
  and format compliance.
- Compares three local language models using validity, stability, speed, judge,
  and factual-alignment measures.
- Presents saved results in an accessible five-page Streamlit dashboard.

## Model roles

- `qwen2.5:3b`: extraction, structured JSON, theme classification, and summaries.
- `llama3.2`: independent evaluation of factual alignment, completeness,
  consistency, and format compliance.
- `phi3:mini`: reserved for the optional three-model extension.

The main extraction and evaluation run uses temperature `0.0` and seed `42`.
The optional stability comparison deliberately uses temperature `0.2` and
varied seeds so run-to-run consistency can be measured. Every prompt is stored
in `src/development_intelligence/prompts.py`.

## Requirements

- Windows with PowerShell
- Python 3.10 or later
- [Ollama](https://ollama.com/download) for running the analysis pipeline
- The source report saved in the project root as `timor-Leste.pdf`

Ollama is not required merely to view the committed dashboard results.

## Setup on Windows

1. Install and launch [Ollama](https://ollama.com/download).
2. Pull the local models:

   ```powershell
   ollama pull qwen2.5:3b
   ollama pull llama3.2
   ollama pull phi3:mini
   ```

3. Create a Python environment and install the project:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   $env:PYTHONPATH = "src"
   ```

## Run the pipeline

The deterministic preparation stage extracts pages, chapter boundaries,
page-aware chunks, raw tables, a keyword theme baseline, and time-series
candidates. It does not require Ollama.

```powershell
python -m development_intelligence.cli prepare
python -m development_intelligence.cli check-models
python -m development_intelligence.cli smoke
python -m development_intelligence.cli run
```

Optional three-model extension (27 generation runs plus independent judging):

```powershell
python -m development_intelligence.cli compare
```

For a partial debugging run after the two-call smoke test:

```powershell
python -m development_intelligence.cli run --limit-chunks 2
```

Do not use a limited run for the final submitted results. Model calls are
cached under `outputs/cache`, so interrupted runs can continue without paying
the full runtime again.

## Launch the interactive dashboard

```powershell
 py -m streamlit run dashboard.py
```

The dashboard has five pages: Overview; Themes; Indicators and trends;
Strengths and challenges; and Model evaluation. The Strengths and challenges
page separates evidence-backed achievements from constraints while retaining
the source PDF pages and extractor-model lineage for every item.

Trend series are organised into related families rather than mixed on one
axis. The chart starts with one series, keeps dates in chronological order,
uses distinct colours and dashed projections, and automatically switches to
small multiples when more than three series are selected. Each chart has an
accessible data-table alternative.

Evidence provenance is retained through page-aware chunks and surfaced in
summaries, indicators, trends, and strengths/challenges so a dashboard claim
can be checked against its PDF page and source excerpt.

The optional comparison reports JSON validity, run-to-run stability, theme and
numerical-fact coverage, missing fields, verbosity, generation speed, and
judge-scored factual alignment. Each model/chunk pair is judged using its first
valid response across all runs; the evaluated run is recorded explicitly.

## Test

```powershell
pytest
```

## Output structure

- `outputs/processed`: deterministic page, section, chunk, and table artifacts.
- `outputs/results`: summaries, themes, indicators, trends, strengths/challenges,
  evaluations, and reproducibility manifests.
- `outputs/cache`: resumable Ollama responses (not committed by default).

## Repository structure

```text
.
|-- dashboard.py                 # Interactive Streamlit dashboard
|-- src/development_intelligence # Extraction and analysis package
|-- tests                        # Automated tests
|-- outputs
|   |-- processed                # Deterministic extraction artifacts
|   |-- results                  # Saved analysis and evaluation results
|   `-- figures                  # Dashboard charts saved as PNG images
|-- timor-Leste.pdf              # Source development report
|-- pyproject.toml
|-- requirements.txt
|-- README.md
`-- .gitignore
```

Every chart displayed by the dashboard is automatically saved to
`outputs/figures/` as a high-resolution PNG image. Charts are generated and
saved when their corresponding dashboard page is opened and rendered; they are
not all created immediately when the dashboard starts. These files display
directly on GitHub and can be committed with the rest of the project.

## Evidence and reproducibility

Outputs retain PDF page references and, where applicable, source excerpts and
model lineage. Raw model responses are preserved alongside validated results so
that parsing and evaluation decisions can be audited. Run settings and model
metadata are recorded in `outputs/results/run_manifest.json`.

## Data source

United Nations Development Programme, *Timor-Leste National Human Development
Report 2018: Planning the Opportunities for a Youthful Population*.

