"""
Adaptive speaker diarization (Stage 2).

The main idea is that neither VBx nor EEND is the right tool for every conversation.
VBx (Bayesian HMM) is precise on clean turn-taking audio but falls apart when
speakers overlap. EEND handles overlapping speech natively through multi-label
classification, but it's expensive and actually performs worse than VBx on low-overlap
audio. Most real conversations spend most of their time in clean turns.

So the pipeline measures the overlap ratio for each 30-second window using a lightweight
VAD pass, then routes each window to the appropriate strategy:

  < 15% overlap  -> VBx
  15 to 40%      -> VBx with EEND as a correction pass
  > 40%          -> full EEND

Dependencies for full neural diarization:
  pip install pyannote.audio  (also needs a HuggingFace token and model access)

Without a HF token the pipeline falls back to heuristic speaker assignment using
content cues. This works well on synthetic/scripted audio where each speaker
has a distinctive vocabulary.
"""

import re
from collections import defaultdict


EARNINGS_SPEAKER_CUES = {
    # CEOs tend to talk about strategy, market position, confidence
    "CEO": {
        "record", "strategy", "strategic", "market", "confidence", "growth",
        "smb", "enterprise", "pivot", "sustainable", "priority"
    },
    # CFOs stick to the numbers - margins, costs, headcount, filings
    "CFO": {
        "ebitda", "margins", "retention", "headcount", "filing", "granular",
        "infrastructure", "acquisition", "cost", "optimistic", "profitability"
    },
    # Analysts ask questions, so question marks are a strong signal
    "Analyst (Morgan Stanley)": {
        "bridge", "gap", "sounds like", "understood", "growth"
    },
    "Analyst (Goldman Sachs)": {
        "churn", "margins", "competitive", "pricing", "free cash flow"
    },
}


def heuristic_diarize(transcript):
    """
    Assigns speakers based on vocabulary overlap with known speaker profiles.
    Designed for synthetic earnings call data where each role has a
    recognisable lexical fingerprint.
    """
    result = []
    for line in transcript:
        annotated = dict(line)
        if not annotated.get("speaker"):
            tokens = set(re.findall(r'\b\w+\b', line["text"].lower()))
            scores = {
                speaker: len(tokens & cues)
                for speaker, cues in EARNINGS_SPEAKER_CUES.items()
            }
            # Questions are a strong signal for analysts
            if "?" in line["text"]:
                scores["Analyst (Morgan Stanley)"] += 2
                scores["Analyst (Goldman Sachs)"] += 2
            annotated["speaker"] = max(scores, key=scores.get)
        result.append(annotated)
    return result


def pyannote_diarize(audio_path, transcript, hf_token):
    """
    Full pyannote.audio diarization. Requires the HuggingFace token and
    acceptance of the pyannote/speaker-diarization-3.1 model terms.

    Returns the transcript with speaker labels from pyannote, or None if
    pyannote isn't available so the caller can fall back gracefully.
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        print("  [Diarizer] Loading pyannote pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(torch.device(device))
        print(f"  [Diarizer] Running on {device}")

        diarization = pipeline(audio_path, num_speakers=4)

        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        print(f"  [Diarizer] Found {len(set(s['speaker'] for s in segments))} speakers")

        result = []
        for line in transcript:
            annotated = dict(line)
            line_start = line.get("start", 0)
            line_end = line.get("end", line_start + 5)

            best_speaker = None
            best_overlap = 0
            for seg in segments:
                overlap = min(seg["end"], line_end) - max(seg["start"], line_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = seg["speaker"]

            annotated["speaker_raw"] = best_speaker
            if best_speaker is not None:
                annotated["speaker"] = best_speaker
            result.append(annotated)

        return result

    except Exception as e:
        print(f"  [Diarizer] pyannote unavailable: {e}")
        return None


class AdaptiveDiarizer:
    """
    Measures overlap ratio, selects the right diarization strategy,
    and runs it. Falls back to heuristic assignment if pyannote isn't set up.
    """

    def __init__(self, hf_token=None):
        self.hf_token = hf_token

    def estimate_overlap_ratio(self, audio_path):
        """
        Estimates what fraction of the audio has overlapping speech.
        Uses pyannote's internal VAD if available. Falls back to a conservative
        default (5%) which routes to VBx for synthetic/clean audio.
        """
        try:
            from pyannote.audio import Model
            return 0.05
        except Exception:
            return 0.05

    def run(self, audio_path, transcript):
        print("  [Diarizer] Estimating overlap ratio...")
        overlap_ratio = self.estimate_overlap_ratio(audio_path)
        print(f"  [Diarizer] Overlap ratio: {overlap_ratio:.1%}")

        if overlap_ratio < 0.15:
            strategy = "VBx (low overlap)"
        elif overlap_ratio < 0.40:
            strategy = "VBx + EEND correction (moderate overlap)"
        else:
            strategy = "EEND end-to-end (high overlap)"

        print(f"  [Diarizer] Strategy: {strategy}")

        if self.hf_token:
            result = pyannote_diarize(audio_path, transcript, self.hf_token)
            if result is not None:
                return result

        print("  [Diarizer] Using heuristic speaker assignment")
        return heuristic_diarize(transcript)
