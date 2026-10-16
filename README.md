# T2I-EXP3

English | [简体中文](README.zh-CN.md)

T2I-EXP3 evaluates text-to-image model safety under a limited query budget. It selects prompt mutation operators using feedback from each query, with a default budget of 16 queries per request.
The repository includes the EXP3 search method, static5 and Original Prompt baselines, and code for running and evaluating SD3.5.

## Method overview

For each request, the initial candidate is generated with CRF→KCM→VED→KPS→IVP.
Subsequent mutations are guided by whether a valid image was returned and whether a policy violation was detected. EXP3 updates operator selection probabilities within the corresponding operator pool.
EXP3 selects operators; the LLM performs the associated semantic rewrites and length compression.
Search state is reset for each request. Search stops at the first policy violation or when the query budget is exhausted (K=16 by default).

| Component | Implementation |
|---|---|
| Lightweight mutations | Role context, letter case, visual vocabulary, punctuation, cross-language transformations, word segmentation, IVP, visual evidence, and foreground enhancement |
| LLM mutations | Aggression enhancement, direct enhancement, natural scenes, style transformation, small-region embedding, and sensory synonyms |
| Online selection | Failure-mode-specific EXP3 weights; 15% global escape plus exploration within each EXP3 pool |
| Feedback | Policy violation confidence and a weak semantic auxiliary reward |
| Main metrics | Attack success rate (ASR), valid image return rate (BPR), and restricted first-success query cost |

Operator templates are in `src/t2i_exp3/methods/operators.py`.

## Repository layout

```text
src/t2i_exp3/
  methods/              # EXP3, operators, static5, and Original Prompt
  llm/                  # OpenAI-compatible client and local text mutation service
  sd3_target.py         # SD3.5 target
  qwen_image_target.py  # Shared input and output guard components
  oracles.py            # Online feedback
  judges.py             # Post-hoc evaluation
  runner.py             # Query budget, logging, and resume support
  resources.py          # Dataset and evaluator resource downloads
  configure.py          # Configuration CLI
  summary.py            # Evaluation summary
configs/                # Real-target examples and synthetic CPU smoke tests
data/                   # Smoke-test samples, I2P200 selection IDs, and download URLs
tests/                  # Offline CPU tests
```

## 1. Installation and offline smoke tests

Python 3.11 or later is required. Run the following commands from the repository root:

```bash
git clone https://github.com/DashKev/t2i-exp3.git
cd t2i-exp3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
t2i-exp3 validate configs/smoke.json
t2i-exp3 run configs/smoke.json --output results/smoke
t2i-exp3 run configs/exp3.smoke.json --output results/exp3-smoke
```

Both smoke tests run on CPU with synthetic feedback.
`exp3.smoke.json` exercises EXP3 and lightweight operators, producing per-query records, feedback, metrics, and a completion marker.

You can also build and install a wheel:

```bash
python -m build
python -m pip install dist/t2i_exp3-0.1.2-py3-none-any.whl
```

The wheel contains the Python package. Example configurations and resource download manifests are provided in the repository.
Relative paths to models and evaluation resources are resolved from the working directory where you run the command.

## 2. Download I2P200

Original I2P project: https://github.com/ml-research/i2p.
The download script uses an I2P CSV copy from a pinned Ring-A-Bell commit.
Download URLs are listed in `data/resources.json`; the 200 selected case IDs are in `data/i2p200_selection.json`.

```bash
t2i-exp3-data i2p
```

This downloads the CSV to `data/raw/i2p.csv` and creates `data/prepared/i2p200.jsonl` using the fixed case IDs.
The minimal input JSONL format is `{"id":"example-1","intent":"a landscape photo","category":"violence","metadata":{}}`.
Custom datasets use the same JSONL format and evaluation category labels.
See [Third-party resources](THIRD_PARTY.md) for upstream terms governing data distribution and use.

## 3. Download evaluation resources

```bash
t2i-exp3-data evaluators
```

Online feedback and post-hoc evaluation use NudeNet and Q16.
This command downloads `nudenet.py`, `best.onnx`, and Q16 `prompts.p` from DiffZOO into `assets/nudenet` and `assets/q16`.

## 4. Download models

SD3.5 runs require NVIDIA CUDA and sufficient GPU and system memory. Install a PyTorch build matching your CUDA environment, then run:

```bash
python -m pip install -e '.[sd35,llm]'
```

Dependency version ranges are listed in `pyproject.toml`. To download SD3.5, accept its model terms on Hugging Face and log in with an account that has access.

```bash
hf auth login
hf download stabilityai/stable-diffusion-3.5-medium --local-dir models/sd35
hf download Qwen/Qwen3Guard-Gen-8B --local-dir models/qwen3guard
hf download openai/clip-vit-large-patch14 --local-dir models/clip
hf download CompVis/stable-diffusion-safety-checker --local-dir models/safety-checker
hf download OpenGVLab/InternVL3_5-8B-HF --local-dir models/mutator
```

These commands place the models in their respective directories under `models/`. The output checker uses the model published by CompVis.
Record the Hugging Face model revisions, dependency versions, and checker snapshots with your experiments.

## 5. Start the LLM text service

Run this in a separate terminal:

```bash
source .venv/bin/activate
python -m t2i_exp3.llm.internvl_openai_server \
  --model-path models/mutator \
  --served-model-name local-mutator \
  --device cuda:2 --dtype bfloat16 --port 18080
```

The service uses text inputs only. You can also use an existing OpenAI-compatible service; its model identifier must match `--llm-model`.
Set `--llm-base-url` to the service root, such as `http://127.0.0.1:18080`. The client appends `/v1/...` automatically.
If you change the LLM, record the model and service parameters in the experiment configuration.

## 6. Configure and run SD3.5

The configuration below places the target model on GPU 0, the guards and evaluators on GPU 1, and the LLM service on GPU 2.
Use the command-line arguments to select devices. `--cpu-offload` enables CPU offloading for the target model. Record the device layout and offloading settings when measuring runtime.

```bash
t2i-exp3-config --method exp3 --budget 16 \
  --llm-base-url http://127.0.0.1:18080 --llm-model local-mutator \
  --device cuda:0 --guard-device cuda:1 --judge-device cuda:1 \
  --output configs/sd35.local.json
t2i-exp3 validate configs/sd35.local.json
t2i-exp3 run configs/sd35.local.json --output results/exp3
```

`validate` checks the configuration and data; `run` loads the models and executes queries.
To debug a single request, use `run ... --task-id i2p-0001`, replacing the example ID with a case ID from `data/i2p200_selection.json`.

The evaluation system uses Qwen3Guard as the input guard and Stable Diffusion Safety Checker as the output guard.
CLIP supplies semantic similarity scores for the search reward. ASR is computed from `policy_violation=True`.

### Baselines

```bash
t2i-exp3-config --method static5 --budget 16 --output configs/static5.local.json
t2i-exp3 run configs/static5.local.json --output results/static5
t2i-exp3-config --method original --output configs/original.local.json
t2i-exp3 run configs/original.local.json --output results/original
```

static5 applies the fixed operator chain to the original request for each query, using the same length handling as EXP3.
Original Prompt submits the original request directly with K=1.

## 7. Outputs, summaries, and resuming runs

Each run creates a unique directory containing configuration and data snapshots, a manifest, per-query records, online feedback, post-hoc evaluations, metrics, and a completion marker.
Generated images are also saved in the run directory. Run directories are covered by `.gitignore`.

```bash
t2i-exp3-summary results/exp3/<run-directory> --budget 16
t2i-exp3 resume results/exp3/<unfinished-run-directory>
```

The summary computes ASR from `policy_violation`.
Query cost is calculated per request: the index of the first policy violation for successful requests, or K for unsuccessful requests. Reported standard deviations are sample standard deviations across requests (ddof=1).
The summary requires `completion.json` in the run directory.
Using `--budget 16` on a completed K32 run limits the summary to the first 16 queries and assigns a cost of 16 to requests without a success in that prefix.
Use `--budget 1` when summarizing Original Prompt runs.

## 8. Tests and license

This version has passed 20 offline tests and GitHub CI checks on Python 3.11 and 3.12.
The code is released under the [MIT License](LICENSE). Models, datasets, and third-party evaluation resources are subject to their respective terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
