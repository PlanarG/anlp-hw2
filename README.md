# HW2: Retrieval-Augmented QA over Pittsburgh and CMU

A retrieval-augmented question-answering system for CMU 11-711 Homework 2. It answers factual
questions about Pittsburgh and Carnegie Mellon (history, culture, events, sports, taxes and
budgets, museum collections) from a fixed local corpus, using two locally served open models.

## How it works

1. **Corpus**: 8,534 documents split into 26,574 chunks (≤180 words, 30-word overlap), stored as JSONL
   under `data/corpus/`.
2. **Retrieval**: hybrid search over the chunks.
   - Sparse: BM25 (k1 = 1.5, b = 0.75) with date-alias tokenization.
   - Dense: [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) (Ollama `qwen3-embedding:8b`,
     Q4_K_M), normalized vectors in a FAISS inner-product index; queries use an instruction prefix.
   - Weighted fusion (sparse 2.0, dense 1.0) over 50 candidates, at most 2 chunks per document, top 6,
     plus ±1 neighboring chunk. The `record_safe` profile adds field-level views for structured records
     (museum catalog entries, schedules, tax tables) so the reader sees the exact record a question names.
3. **Reading**: [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
   (Ollama `qwen3:30b-a3b-instruct-2507-q4_K_M`) answers from at most 14,000 characters of evidence with
   temperature 0, returning strict JSON (`{"answer": ...}`) or `UNKNOWN`.
4. **Answer normalization**: deterministic, whole-answer formatting rules (`src/hw2rag/qa/normalize.py`).
   They never add facts, and every applied rule is recorded in the run trace.

All run settings, including both models' Ollama digests and the prompt hash, are in `configs/pipeline.json`.

## Repository layout

```
configs/pipeline.json        models (pinned by digest), prompt hash, retrieval/reader parameters, paths
data/corpus/documents/       processed documents (JSONL per source) + _manifest.json
data/corpus/chunks/          chunks used for indexing (JSONL per source) + _manifest.json
hw2-public/                  question sets: leaderboard_queries.json, test_set_day_2.json
src/hw2rag/indexing/         chunking (chunk.py, build_chunks.py) and index building (build_index.py)
src/hw2rag/retrieval/        BM25, dense index, hybrid fusion, record-safe evidence, search
src/hw2rag/qa/               prompt, reader, generation loop, answer normalization, validation, evaluation
scripts/run_pipeline.py      end-to-end runner (starts the Ollama workers, builds the index, answers)
scripts/serve_models.py      starts one project-local Ollama worker
scripts/calibrate_outputs.py re-applies the calibrated normalization to an existing run trace
system_outputs/              submitted outputs for test_set_day_2.json
outputs/                     run outputs with full traces (evidence, raw reader output, applied rules)
```

The web collection and HTML/PDF extraction code that produced the corpus is not part of this
repository; the corpus is included in processed form.

## Corpus

| Group | Sources | Documents | Chunks |
|---|---|---:|---:|
| Original collection: city tax pages and 2025 operating budget (PDF), Carnegie Museum of Art catalog, CMU/Downtown/Pittsburgh event calendars, Pirates/Penguins/Steelers rosters and schedules, college sports, CMU history and research, culture, food and festival guides | 29 | 5,531 | 8,089 |
| Wikipedia, by category: Pittsburgh landmarks (buildings, bridges, parks, museums, streets, neighborhoods), history and culture (biographies excluded), sports, CMU and other universities | 4 | 1,496 | 12,696 |
| Official sites, ~16 pages each selected by generic keywords (about/history/visit/contact/events/...): CMU, other universities and athletics, museums, performing arts, venues, sports, festivals, city/county/visitor sites | 8 | 1,507 | 5,789 |

Per-source counts are in `data/corpus/documents/_manifest.json` and `data/corpus/chunks/_manifest.json`.

## Setup

- Linux x86_64, Python 3.12+, an NVIDIA GPU (about 6.5 GB free for the embedding model and 22 GB for the
  reader; they can be on different GPUs).
- `python -m pip install -r requirements.txt`
- [Ollama](https://github.com/ollama/ollama) 0.40.11 on `PATH`, with both models pulled into a model store:

  ```bash
  ollama pull qwen3-embedding:8b                    # digest 64b933495768...
  ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M    # digest 19e422b02313...
  ```

  The runner checks both digests against `configs/pipeline.json` and stops if they differ. The dense index
  is tied to the embedding model, so the encoder must not be swapped.
- The official Ollama 0.40.11 Linux build needs NVIDIA driver ≥ 550. On older drivers (e.g. 535 / CUDA 12.2)
  it falls back to CPU (its log says "NVIDIA driver too old"), which the runner refuses. In that case build the same Ollama version from
  source against the local CUDA toolkit:

  ```bash
  git clone --depth 1 --branch v0.40.11 https://github.com/ollama/ollama.git && cd ollama
  cmake -B build -G Ninja . -DOLLAMA_LLAMA_BACKENDS=cuda_v12 -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCUDAToolkit_ROOT=/usr/local/cuda-12.2          # needs Go and CMake >= 3.24
  cmake --build build && cmake --install build --prefix /path/to/ollama-cu122
  ```

## Usage

```bash
# 1. Check that the corpus loads (no models needed)
python scripts/run_pipeline.py --check

# 2. Build the BM25 + dense index into artifacts/indexes/qwen3_8b (about 30-45 min on one A6000 for 26.6k chunks)
python scripts/run_pipeline.py --build-only --models-dir /path/to/ollama/models --embedding-gpu 0

# 3. Answer a question set (system 1 normalization)
python scripts/run_pipeline.py --models-dir /path/to/ollama/models --embedding-gpu 0 --reader-gpu 1 \
    --questions hw2-public/test_set_day_2.json --output outputs/test_set_day_2/predictions.json

#    Leaderboard submission (adds the Andrew ID field)
python scripts/run_pipeline.py --models-dir /path/to/ollama/models --andrewid <andrewid>

# 4. System 2: the same run with the calibrated normalization, either at generation time
#    (--answer-normalization calibrated) or applied to an existing trace:
python scripts/calibrate_outputs.py --trace outputs/test_set_day_2/predictions.trace.json \
    --output system_outputs/system_output_2.json
```

`run_pipeline.py` starts its own Ollama workers on ports 11434 and 11436 and stops only the processes it
started. It stops if a port is busy, if a GPU lacks free memory, or if a model is not fully GPU-resident.
Answers are checkpointed after each question; rerunning the same command resumes. Each run writes
`predictions.json`, `predictions.trace.json` (retrieved evidence, raw reader output, applied rules) and
`validation.json` (format and configuration checks only, no scores).

## System outputs (`test_set_day_2.json`, 250 questions)

Both outputs use the same retrieval and reader. They differ only in answer normalization:

- `system_output_1.json`: the base `fields` normalization (calendar dates without ordinals or weekdays,
  12-hour clock times, acquisition method without donor credit, and similar).
- `system_output_2.json`: `fields` plus the `calibrated` rules. These were derived from an error analysis
  of the public leaderboard questions and are applied identically to every question:
  - machine timestamps such as `2026-11-10T12:30:00-05:00` become the requested clock time or date;
  - bare clock times are written canonically (`7 PM` becomes `7:00 PM`);
  - roster position codes become names when the sport is clear from the question (`K` becomes `Kicker`);
  - for "Who should I contact…" questions, "Contact X for …" sentences are reduced to X, but only when X
    contains an email address or phone number or is at most 8 words;
  - for entity questions (which / who / "what <noun>"), sentences that restate the question
    ("The venue that hosts … is Y.") are reduced to Y, but only when Y is at most 8 words.

  On the test set, these rules change 6 of 250 answers, all of them time formats.

## Leaderboard set (150 questions)

The answers were checked by hand against the corpus; there is no official key, so treat these counts as
approximate. With the original 29-source corpus, about 85 answers were correct. After adding the Wikipedia
and official-site sources, about 115 were correct (UNKNOWN went from 19 to 11).

The remaining errors fall into three groups:
- the answer is missing from the corpus and the reader guesses instead of answering `UNKNOWN`;
- the reader misreads a table that is in the evidence (counting rows, filtering by a column);
- generic questions miss the relevant chunk during retrieval.
