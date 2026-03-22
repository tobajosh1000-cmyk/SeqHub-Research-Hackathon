"""
Stage 1: Transcription.

The strategy is:
  1. Run Whisper locally to get a raw transcript with timestamps
  2. Optionally pass the raw output through Claude to clean up punctuation,
     fix run-on segments, and handle any hallucinated phrases

Note: the Claude API doesn't accept audio files directly, so Whisper handles
the actual speech-to-text step. Claude is used as a post-processing pass
to clean and structure the output before it goes into the pipeline.
"""

import json
import os
import re
import subprocess
import urllib.request


def transcribe_with_whisper_cli(audio_path):
    """
    Tries to transcribe using the openai-whisper Python package first,
    then falls back to the whisper CLI if the package isn't installed.
    Returns a list of dicts with text, start, and end keys.
    Returns None if neither is available.
    """
    # Try the Python package
    try:
        import whisper
        print("  [Transcribe] Using openai-whisper...")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)
        return [
            {"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]}
            for seg in result["segments"]
        ]
    except ImportError:
        pass

    # Try the CLI
    try:
        result = subprocess.run(
            ["whisper", audio_path, "--output_format", "json", "--model", "base"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return [
                {"text": s["text"].strip(), "start": s["start"], "end": s["end"]}
                for s in data["segments"]
            ]
    except Exception:
        pass

    return None


def clean_transcript_with_claude(raw_lines):
    """
    Sends the raw Whisper output to Claude to fix punctuation, merge
    run-on segments, and remove obvious hallucinations.

    Returns the cleaned lines, or the original lines if the API call fails.
    """
    try:
        raw_text = "\n".join(
            f"[{l.get('start', i * 5):.1f}s] {l['text']}"
            for i, l in enumerate(raw_lines)
        )

        prompt = (
            "You are cleaning a raw ASR transcript from an earnings call.\n\n"
            "The text below may have missing punctuation, run-on segments, or "
            "repetitive phrase hallucinations.\n"
            "Clean it and identify natural speaker turn boundaries.\n\n"
            "Return ONLY a JSON array. Each object must have:\n"
            "- \"text\": the cleaned utterance\n"
            "- \"start\": timestamp from the input (preserve exactly)\n"
            "- \"end\": timestamp from the input (preserve exactly)\n\n"
            f"Raw transcript:\n{raw_text}"
        )

        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        text = data["content"][0]["text"].strip()
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        return json.loads(text)

    except Exception as e:
        print(f"  [Transcribe] Claude cleaning skipped: {e}")
        return raw_lines


def transcribe_with_claude(audio_path):
    """
    Main entry point for Stage 1.
    Runs Whisper, then optionally cleans with Claude.
    Raises if no transcription backend is found.
    """
    print(f"  [Transcribe] Processing: {audio_path}")

    raw_lines = transcribe_with_whisper_cli(audio_path)

    if raw_lines is None:
        raise RuntimeError(
            "No transcription backend found. Install one of:\n"
            "  pip install openai-whisper\n"
            "  pip install whisperx\n"
            "Or use --gt-mode to skip ASR and test the downstream stages directly."
        )

    print(f"  [Transcribe] Whisper produced {len(raw_lines)} segments")

    result = []
    for i, line in enumerate(raw_lines):
        result.append({
            "line_index": i,
            "text": line["text"],
            "start": line.get("start", i * 5.0),
            "end": line.get("end", i * 5.0 + 4.9),
        })

    return result
