"""
SeqHub AI Research Hackathon - Research Direction 02
Adaptive Pipeline for Speaker Diarization and Hierarchical Topic Segmentation

Five stages, each independently evaluable:
  1. Transcription     -> WER
  2. Speaker Diarization -> DER
  3. Topic Segmentation  -> NMI, Cv
  4. Boundary Detection  -> WindowDiff, Pk
  5. Sentiment Analysis  -> Macro-F1
"""

import json
import os
import sys
from pathlib import Path

from topic_segmentation import TopicSegmenter
from sentiment import SentimentClassifier
from evaluation import Evaluator


class EarningsCallPipeline:
    """
    Runs the full pipeline on a single earnings call audio file.
    Each stage is a separate module so failures are easy to isolate.
    """

    def __init__(self, use_api_transcription=True, hf_token=None):
        self.use_api_transcription = use_api_transcription
        self.hf_token = hf_token
        self.topic_segmenter = TopicSegmenter()
        self.sentiment_classifier = SentimentClassifier()

    def transcribe(self, audio_path, transcript_override=None):
        """
        Stage 1. Returns a list of dicts with speaker, text, start, end keys.

        Pass transcript_override to skip ASR entirely - useful for testing
        the downstream stages in isolation without needing the audio.
        """
        if transcript_override is not None:
            print("[Stage 1] Skipping ASR, using provided transcript")
            return transcript_override

        if self.use_api_transcription:
            from transcribe_api import transcribe_with_claude
            return transcribe_with_claude(audio_path)
        else:
            from transcribe_whisper import transcribe_with_whisper
            return transcribe_with_whisper(audio_path, self.hf_token)

    def diarize(self, audio_path, transcript):
        """
        Stage 2. Adaptive routing between VBx and EEND based on measured overlap.
        Returns transcript lines with speaker labels filled in.
        """
        from diarization import AdaptiveDiarizer
        diarizer = AdaptiveDiarizer(hf_token=self.hf_token)
        return diarizer.run(audio_path, transcript)

    def segment_topics(self, transcript):
        """
        Stages 3 and 4. Assigns topic labels and marks segment boundaries.
        """
        return self.topic_segmenter.run(transcript)

    def analyze_sentiment(self, transcript):
        """
        Stage 5. Adds sentiment label per speaker turn.
        """
        return self.sentiment_classifier.run(transcript)

    def run(self, audio_path, transcript_override=None):
        print("\n" + "="*55)
        print("SeqHub Pipeline - Starting")
        print("="*55 + "\n")

        transcript = self.transcribe(audio_path, transcript_override)
        print(f"[Stage 1] Done - {len(transcript)} lines\n")

        transcript = self.diarize(audio_path, transcript)
        print("[Stage 2] Done\n")

        transcript = self.segment_topics(transcript)
        print("[Stage 3/4] Done\n")

        transcript = self.analyze_sentiment(transcript)
        print("[Stage 5] Done\n")

        return {"lines": transcript}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SeqHub Earnings Call Pipeline")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--ground-truth", help="Path to ground truth JSON")
    parser.add_argument("--output", default="outputs/pipeline_output.json")
    parser.add_argument("--gt-mode", action="store_true",
                        help="Use ground truth transcript text and skip ASR. Good for testing topic/sentiment stages in isolation.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    ground_truth = None
    gt_transcript = None

    if args.ground_truth:
        with open(args.ground_truth) as f:
            ground_truth = json.load(f)
        if args.gt_mode:
            gt_transcript = [
                {"line_index": l["line_index"], "speaker": l["speaker"], "text": l["text"]}
                for l in ground_truth["lines"]
            ]

    pipeline = EarningsCallPipeline(
        use_api_transcription=True,
        hf_token=args.hf_token
    )

    result = pipeline.run(args.audio, transcript_override=gt_transcript)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Output] Saved to {args.output}")

    if ground_truth:
        evaluator = Evaluator()
        scores = evaluator.evaluate(result["lines"], ground_truth["lines"])
        print("\n" + "="*55)
        print("EVALUATION RESULTS")
        print("="*55)
        for metric, score in scores.items():
            print(f"  {metric:<25} {score:.4f}")

        scores_path = args.output.replace(".json", "_scores.json")
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\n[Scores] Saved to {scores_path}")


if __name__ == "__main__":
    main()
