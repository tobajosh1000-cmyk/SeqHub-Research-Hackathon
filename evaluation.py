"""
Evaluation module - computes all seven hackathon metrics.

Each metric is implemented from the spec directly, no external scoring libraries.

  WER         Word Error Rate         - transcription quality
  DER         Diarization Error Rate  - speaker attribution
  NMI         Normalized Mutual Info  - topic cluster alignment
  Cv          Topic Coherence         - intra-topic keyword coherence
  WindowDiff  Boundary detection      - stricter, counts exact boundary discrepancies
  Pk          Boundary detection      - slightly more lenient than WindowDiff
  Macro-F1    Sentiment               - balanced across all sentiment classes
"""

import math
import re
from collections import Counter, defaultdict


# WER - Word Error Rate

def compute_wer(hypothesis, reference):
    """
    Standard edit distance (substitutions, deletions, insertions) normalised
    by the reference length. Uses dynamic programming.
    """
    hyp = hypothesis
    ref = reference
    n, m = len(ref), len(hyp)

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])

    return dp[m] / max(n, 1)


def wer_from_transcripts(predicted_lines, gt_lines):
    pred_words = " ".join(l["text"] for l in predicted_lines).lower().split()
    ref_words  = " ".join(l["text"] for l in gt_lines).lower().split()
    return compute_wer(pred_words, ref_words)


# DER - Diarization Error Rate

def compute_der(predicted_lines, gt_lines):
    """
    Simplified line-level speaker error rate. True DER needs time-segment
    alignment; this uses line-level speaker accuracy as a proxy, which is
    appropriate when lines and speaker turns correspond 1-to-1.
    """
    if len(predicted_lines) != len(gt_lines):
        n = min(len(predicted_lines), len(gt_lines))
        predicted_lines = predicted_lines[:n]
        gt_lines = gt_lines[:n]

    errors = sum(
        1 for p, g in zip(predicted_lines, gt_lines)
        if p.get("speaker", "").strip() != g.get("speaker", "").strip()
    )
    return errors / max(len(gt_lines), 1)


# NMI - Normalized Mutual Information

def entropy(labels):
    n = len(labels)
    counts = Counter(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def compute_nmi(predicted_topics, gt_topics):
    """
    Measures how well the predicted topic clusters align with the ground truth.
    Score of 1.0 is perfect alignment, 0 is no better than random.
    """
    n = len(predicted_topics)
    assert n == len(gt_topics), "Length mismatch between predicted and ground truth"

    joint = Counter(zip(predicted_topics, gt_topics))
    h_pred = entropy(predicted_topics)
    h_gt   = entropy(gt_topics)

    mi = 0.0
    for (p, g), count in joint.items():
        p_pred  = Counter(predicted_topics)[p] / n
        p_gt    = Counter(gt_topics)[g] / n
        p_joint = count / n
        if p_joint > 0:
            mi += p_joint * math.log2(p_joint / (p_pred * p_gt))

    denom = (h_pred + h_gt) / 2
    return mi / denom if denom > 0 else 0.0


# Cv - Topic Coherence (NPMI-based)

def tokenize(text):
    return re.findall(r'\b[a-z]{3,}\b', text.lower())


def compute_cv_coherence(predicted_lines, top_n=5):
    """
    For each topic, finds the top-N words by frequency, then computes pairwise
    NPMI using within-topic document co-occurrence as the corpus.

    This has a known floor on short segments - NPMI needs enough co-occurrence
    to produce meaningful scores. With only 2-4 utterances per topic the scores
    will be low regardless of how coherent the topics actually are.
    """
    topic_docs = defaultdict(list)
    for line in predicted_lines:
        topic = line.get("topic", "Unknown")
        topic_docs[topic].append(line["text"])

    if len(topic_docs) <= 1:
        return 0.0

    all_scores = []

    for topic, docs in topic_docs.items():
        if len(docs) < 2:
            continue

        doc_tokens = [tokenize(d) for d in docs]
        all_tokens = [t for toks in doc_tokens for t in toks]

        if not all_tokens:
            continue

        top_words = [w for w, _ in Counter(all_tokens).most_common(top_n)]
        if len(top_words) < 2:
            continue

        n_docs = len(doc_tokens)
        word_doc_freq = defaultdict(int)
        pair_doc_freq = defaultdict(int)

        for toks in doc_tokens:
            tok_set = set(toks)
            for w in top_words:
                if w in tok_set:
                    word_doc_freq[w] += 1
            for i, w1 in enumerate(top_words):
                for w2 in top_words[i + 1:]:
                    if w1 in tok_set and w2 in tok_set:
                        pair_doc_freq[(w1, w2)] += 1

        pair_scores = []
        for i, w1 in enumerate(top_words):
            for w2 in top_words[i + 1:]:
                p_w1   = word_doc_freq[w1] / n_docs
                p_w2   = word_doc_freq[w2] / n_docs
                p_pair = pair_doc_freq[(w1, w2)] / n_docs

                if p_pair == 0 or p_w1 == 0 or p_w2 == 0:
                    pair_scores.append(-1.0)
                else:
                    pmi   = math.log2(p_pair / (p_w1 * p_w2))
                    log_p = -math.log2(p_pair)
                    npmi  = pmi / log_p if log_p != 0 else 0.0
                    pair_scores.append(npmi)

        if pair_scores:
            all_scores.append(sum(pair_scores) / len(pair_scores))

    return sum(all_scores) / len(all_scores) if all_scores else 0.0


# WindowDiff and Pk - Topic Boundary Detection

def topics_to_boundaries(topics):
    """Converts a topic sequence to a binary list where 1 marks the start of a new topic."""
    boundaries = [0] * len(topics)
    for i in range(1, len(topics)):
        if topics[i] != topics[i - 1]:
            boundaries[i] = 1
    return boundaries


def compute_pk(pred_topics, gt_topics):
    """
    Probability that two positions at distance k are inconsistently classified
    relative to the ground truth. k is set to half the average segment length.

    Pk is slightly lenient - it partially rewards near-miss boundaries.
    Use WindowDiff if you care more about exact boundary placement.
    """
    pred_b = topics_to_boundaries(pred_topics)
    gt_b   = topics_to_boundaries(gt_topics)
    n = len(gt_b)

    gt_seg_count = sum(gt_b) + 1
    k = max(1, n // (2 * gt_seg_count))

    errors = 0
    total  = 0
    for i in range(n - k):
        gt_same   = (sum(gt_b[i + 1:i + k + 1]) == 0)
        pred_same = (sum(pred_b[i + 1:i + k + 1]) == 0)
        if gt_same != pred_same:
            errors += 1
        total += 1

    return errors / max(total, 1)


def compute_windowdiff(pred_topics, gt_topics):
    """
    Penalises any difference in boundary count within a sliding window.
    More conservative than Pk because near-miss boundaries are still penalised.
    Better for retrieval use cases where an off-by-one boundary means the
    user gets the wrong segment back.
    """
    pred_b = topics_to_boundaries(pred_topics)
    gt_b   = topics_to_boundaries(gt_topics)
    n = len(gt_b)

    gt_seg_count = sum(gt_b) + 1
    k = max(1, n // (2 * gt_seg_count))

    errors = 0
    total  = 0
    for i in range(n - k):
        gt_count   = sum(gt_b[i + 1:i + k + 1])
        pred_count = sum(pred_b[i + 1:i + k + 1])
        if gt_count != pred_count:
            errors += 1
        total += 1

    return errors / max(total, 1)


# Macro-F1 - Sentiment Analysis

def compute_macro_f1(pred_sentiments, gt_sentiments):
    """
    Computes F1 per class then averages equally across all classes.
    This prevents a model that always predicts the dominant class from
    scoring well just because one label dominates the dataset.
    """
    classes = list(set(gt_sentiments))
    f1s = []

    for cls in classes:
        tp = sum(1 for p, g in zip(pred_sentiments, gt_sentiments) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(pred_sentiments, gt_sentiments) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred_sentiments, gt_sentiments) if p != cls and g == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s) if f1s else 0.0


# Master evaluator

class Evaluator:

    def evaluate(self, predicted_lines, gt_lines):
        n    = min(len(predicted_lines), len(gt_lines))
        pred = predicted_lines[:n]
        gt   = gt_lines[:n]

        scores = {}
        scores["WER"] = wer_from_transcripts(pred, gt)
        scores["DER"] = compute_der(pred, gt)

        pred_topics = [l.get("topic", "") for l in pred]
        gt_topics   = [l.get("topic", "") for l in gt]

        scores["NMI"]        = compute_nmi(pred_topics, gt_topics)
        scores["Cv"]         = compute_cv_coherence(pred)
        scores["WindowDiff"] = compute_windowdiff(pred_topics, gt_topics)
        scores["Pk"]         = compute_pk(pred_topics, gt_topics)

        pred_sentiments = [l.get("sentiment", "neutral") for l in pred]
        gt_sentiments   = [l.get("sentiment", "neutral") for l in gt]
        scores["Macro_F1"] = compute_macro_f1(pred_sentiments, gt_sentiments)

        return scores


if __name__ == "__main__":
    import json, sys

    if len(sys.argv) < 2:
        print("Usage: python evaluation.py ground_truth.json [predicted.json]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        gt = json.load(f)["lines"]

    if len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            pred = json.load(f)["lines"]
    else:
        pred = gt
        print("Self-test mode: scoring GT against itself (should be perfect)")

    ev = Evaluator()
    scores = ev.evaluate(pred, gt)

    print("\nResults:")
    for metric, score in scores.items():
        direction = "lower = better" if metric in ("WER", "DER", "WindowDiff", "Pk") else "higher = better"
        print(f"  {metric:<15} {score:.4f}  ({direction})")
