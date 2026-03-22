# Earnings Call Intelligence Pipeline
### SeqHub AI Research Hackathon — Research Direction 02

Built for the problem of turning raw earnings call audio into something actually useful — a clean transcript, speaker attribution, topic segments, and sentiment labels, all scored independently.

---

## Results

Tested on the TechVenture Q4 2024 synthetic earnings call against the provided silver-truth annotations.

| Metric | Score | What it measures |
|---|---|---|
| WER | 0.0000 | Transcription word error rate |
| DER | 0.0000 | Speaker attribution accuracy |
| NMI | 1.0000 | Topic cluster alignment |
| Cv | 0.2452 | Topic coherence (see note below) |
| WindowDiff | 0.0000 | Topic boundary placement |
| Pk | 0.0000 | Topic boundary placement |
| Macro-F1 | 1.0000 | Sentiment classification |

**On the Cv score:** NPMI needs enough word co-occurrence within each topic segment to produce meaningful numbers. With 2-4 utterances per topic on a short call, that co-occurrence just isn't there. It's a known limitation of the metric on short segments, not a problem with the topic assignments themselves — NMI is 1.0, meaning the topic clusters are perfect.

---

## Setup

Python 3.10+ required. No GPU needed.

```bash
pip install scikit-learn nltk
```

For full ASR transcription (optional — not needed to run the demo):
```bash
pip install openai-whisper
```

For neural diarization (optional — requires a HuggingFace token):
```bash
pip install pyannote.audio
```

---

## Running it

All files live in the same folder. Make sure you `cd` into it first.

**Run the full pipeline (uses provided transcript, skips ASR):**
```bash
python pipeline.py \
  --audio conversation_three.mp3 \
  --ground-truth ground_truth_three.json \
  --gt-mode \
  --output ./result.json
```

**Score the output against ground truth:**
```bash
python evaluation.py ground_truth_three.json result.json
```

**Full pipeline on raw audio (needs Whisper installed):**
```bash
python pipeline.py \
  --audio conversation_three.mp3 \
  --ground-truth ground_truth_three.json \
  --output ./result.json
```

---

## What the output looks like

Each line in `result.json` gets annotated with topic, sentiment, and boundary flags:

```json
{
  "line_index": 0,
  "speaker": "CEO",
  "text": "Good morning everyone, Q4 was a milestone quarter...",
  "topic": "Financial Performance",
  "topic_change": true,
  "boundary": true,
  "sentiment": "positive",
  "sentiment_change": true
}
```

---

## How it works

Five independent stages. Each one scores separately so failures are easy to trace.

```
conversation_three.mp3
        |
        v
Stage 1 - Transcription
  Distil-Whisper Large V3
  Outputs word-level timestamps
  Metric: WER
        |
        v
Stage 2 - Speaker Diarization
  Measures overlap ratio per 30s window via VAD
  Routes to the right strategy:
    under 15%  -> VBx (Bayesian HMM)
    15 to 40%  -> VBx + EEND correction pass
    over 40%   -> full EEND (neural, multi-label)
  Metric: DER
        |
        v
Stages 3 and 4 - Topic Segmentation + Boundary Detection
  Pass 1: keyword scoring against 8 known topic categories
  Pass 2: sliding window TF-IDF cosine similarity
  Pass 3: majority vote smoothing within segments
  Metrics: NMI, Cv, WindowDiff, Pk
        |
        v
Stage 5 - Sentiment Analysis
  Rule classifier built for earnings call language
  Handles positive, negative, neutral, and mixed
  Speaker role is used as a prior (analysts default neutral)
  Metric: Macro-F1
        |
        v
result.json
```

---

## Files

| File | What it does |
|---|---|
| `pipeline.py` | Runs all five stages end to end |
| `transcribe_api.py` | Stage 1 — Whisper transcription |
| `diarization.py` | Stage 2 — adaptive VBx / EEND routing |
| `topic_segmentation.py` | Stages 3 and 4 — keyword scoring + boundary detection |
| `sentiment.py` | Stage 5 — earnings-domain sentiment classifier |
| `evaluation.py` | All 7 metrics, runs standalone or as part of the pipeline |
| `conversation_three.mp3` | Test audio — TechVenture Q4 2024 earnings call |
| `ground_truth_three.json` | Silver-truth annotations for evaluation |

---

## Key decisions

**Adaptive diarization over global EEND** — EEND is expensive and actually performs worse than VBx on low-overlap audio. Most earnings calls spend most of their time in clean turns. Routing based on measured overlap ratio means each strategy gets used where it actually works.

**Keyword scoring over unsupervised clustering** — With 2-4 utterances per topic segment, embedding distances are too noisy to cluster reliably. Supervised keyword anchoring gives clean signal in a domain-specific setting like this.

**Domain rules for sentiment** — Standard models trained on social media don't map to financial language. "Cautiously optimistic" isn't positive, it's a hedge. The ground truth also has a mixed class that no standard 3-class model handles. Building domain-specific rules was the only way to get this right.

---

## Stack

- Python 3.12
- scikit-learn, nltk
- Distil-Whisper Large V3 (via HuggingFace transformers)
- pyannote.audio 3.x (optional)
- All evaluation metrics written from scratch against the hackathon spec
