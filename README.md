# Personality Assessment System

A Streamlit-based web application that administers the Big Five (OCEAN) personality assessment, analyzes results using a fine-tuned local LLM, and generates a downloadable HTML report with behavioral analysis from video data.

---

## What it does

1. Presents the user with 44 BFI (Big Five Inventory) questions
2. Scores responses across 5 personality traits: **Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism**
3. Displays a bar chart of normalized scores (0–100)
4. On demand, generates a full report that includes:
   - LLM-generated personality analysis (structured markdown)
   - Radar chart of the personality profile
   - Temporal emotion snapshots from video (valence/arousal data + thermal/optical frames)
   - Downloadable HTML report

---

## Project Structure

```
├── Ocean_Interface.py        # Main Streamlit app — quiz UI and report trigger
├── OceanModel.py             # Local LLM inference — generates personality analysis
├── ReportGeneration.py       # HTML report builder — radar chart, snapshots, Jinja2
├── ocean_finetune.ipynb      # Fine-tuning notebook — trains Qwen2.5-1.5B with LoRA
├── Datasets/
│   ├── Ocean.json            # Original fine-tuning dataset
│   └── Ocean_HQ.json         # High-quality curated fine-tuning dataset (74 examples)
├── Emotional_Behaviour/
│   ├── video.mp4             # Input video for behavioral analysis
│   ├── arousal.npy           # Arousal signal array (per frame)
│   └── valence.npy           # Valence signal array (per frame)
├── requirements.txt
└── .env                      # (not committed) API keys if using fallback
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the model

The fine-tuned model is **not included in this repo** (too large for GitHub).

**Option A — Use your own trained model:**
- Train using `ocean_finetune.ipynb` on Google Colab
- Save the merged model to Google Drive
- Download or mount it locally
- Set the path in `OceanModel.py`:
```python
MODEL_PATH = "/path/to/ocean_model_merged"
```

**Option B — Use the base model without fine-tuning (zero-shot):**
- Set `MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"` in `OceanModel.py`
- The model will be downloaded automatically from Hugging Face (~3 GB)

### 4. Add behavioral data

Place your files in `Emotional_Behaviour/`:
- `video.mp4` — the session recording
- `arousal.npy` — numpy array of arousal values
- `valence.npy` — numpy array of valence values

### 5. Run the app

```bash
streamlit run Ocean_Interface.py
```

---

## Fine-Tuning

Open `ocean_finetune.ipynb` in Google Colab (recommended: T4 or A100 GPU).

| Detail | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | LoRA (no quantization) |
| Dataset | `Datasets/Ocean_HQ.json` |
| Output | LoRA adapter + optional merged model |
| Merged model size | ~3 GB |

The notebook walks through: loading data → applying LoRA → training → saving adapter → merging into a standalone model → testing inference.

---

## LLM Output Schema

The model is prompted to always return this structure:

```
## Personality Overview
## Key Strengths
## Personality Traits
## Blind Spots
## Work & Career Style
## Key Recommendations
```

---

## Requirements

- Python 3.10+
- CUDA GPU recommended for inference (CPU works but is slow)
- ~4 GB disk space for the model
- ~8 GB VRAM for fine-tuning (with gradient checkpointing)

---

## What is NOT in this repo

| File/Folder | Why excluded |
|---|---|
| `ocean_model_merged/` | ~3 GB — too large for GitHub |
| `Emotional_Behaviour/video.mp4` | Large binary file |
| `.env` | Contains API keys |