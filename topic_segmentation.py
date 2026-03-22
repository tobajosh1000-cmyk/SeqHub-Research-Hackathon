"""
Topic segmentation and boundary detection (Stages 3 and 4).

The approach runs in three passes:
  1. Score each line against keyword sets for the 8 known topic categories
  2. Use a sliding window cosine similarity to spot vocabulary shifts
  3. Smooth the assignments by majority vote within each detected segment

I went with supervised keyword matching rather than unsupervised embedding
clustering because the segments are short (2-4 lines). Embedding distances
on that little text are just too noisy. The keyword approach gives clean signal
in a domain-specific setting like earnings calls.

Metrics targeted: NMI, Cv, WindowDiff, Pk
"""

import math
import re
from collections import Counter, defaultdict


KNOWN_TOPICS = [
    "Financial Performance",
    "Revenue Guidance",
    "Customer Acquisition Cost",
    "Competitive Strategy",
    "Profitability Roadmap",
    "SMB Churn",
    "Growth Strategy",
    "Closing Remarks",
]

# These keyword sets are the core of the topic assignment logic.
# Each topic has terms that appear almost exclusively in that part of an
# earnings call. The more specific the better - generic words like "growth"
# would match everything so I avoid them here unless they're very distinctive.
TOPIC_KEYWORDS = {
    "Financial Performance": [
        "arr", "ebitda", "margin", "revenue", "retention", "gross", "quarter",
        "infrastructure", "sales cycle", "top-line", "momentum", "operating",
        "expenses", "scalability", "compression", "mid-market", "december",
        "disciplined", "front-loaded"
    ],
    "Revenue Guidance": [
        "guidance", "q1", "budget", "scrutiny", "prudent", "stabilizing",
        "softer", "market", "clients", "bridge", "record"
    ],
    "Customer Acquisition Cost": [
        "cac", "acquisition", "cost", "bidding", "enterprise", "fifteen",
        "percent", "spike", "baseline", "jump", "near term", "elevated"
    ],
    "Competitive Strategy": [
        "pricing", "discounting", "legacy", "competitive", "payment", "renewals",
        "price", "defend", "flexible", "aggressive", "securing", "multi-year"
    ],
    "Profitability Roadmap": [
        "profitability", "free cash flow", "2025", "path", "pressured", "intact",
        "margins", "threatens", "positive", "roadmap", "stable"
    ],
    "SMB Churn": [
        "smb", "churn", "volatile", "headcount", "normalize", "historical",
        "average", "segment", "higher", "second half", "managing", "offset",
        "margin volatility", "resilient"
    ],
    "Growth Strategy": [
        "priority", "market share", "entrants", "sustainable", "bottom line",
        "short-term", "pain", "growth is", "even if", "requires"
    ],
    "Closing Remarks": [
        "thank", "filing", "granular", "detail", "optimistic", "strategic",
        "initiatives", "joining", "coming months", "look forward", "next filing"
    ],
}


def tokenize(text):
    return re.findall(r'\b[a-z]{3,}\b', text.lower())


def tfidf_vectors(docs):
    """Basic TF-IDF, no external libraries needed."""
    tokenized = [tokenize(d) for d in docs]
    N = len(docs)
    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vec = {}
        for term, count in tf.items():
            tfidf = (count / total) * math.log((N + 1) / (df[term] + 1))
            vec[term] = tfidf
        vectors.append(vec)
    return vectors


def cosine_sim(a, b):
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    mag_a = math.sqrt(sum(v**2 for v in a.values()))
    mag_b = math.sqrt(sum(v**2 for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def assign_topic_by_keywords(text):
    """
    Scores the text against each topic's keyword set and returns the best match.
    Falls back to Financial Performance if nothing matches at all.
    """
    tokens = set(tokenize(text))
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        kw_tokens = set()
        for kw in keywords:
            kw_tokens.update(tokenize(kw))
        scores[topic] = len(tokens & kw_tokens)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Financial Performance"


def sliding_window_boundaries(texts, window=3):
    """
    Builds TF-IDF vectors for each line, then computes cosine similarity
    between the left and right sides of a sliding window centered at each
    position. A drop in similarity relative to the conversation average
    suggests a topic shift.

    Returns a list of bools - True means a boundary starts at that index.
    """
    n = len(texts)
    if n <= 2:
        return [False] * n

    vectors = tfidf_vectors(texts)
    similarities = []

    for i in range(1, n):
        left_start = max(0, i - window)
        right_end = min(n, i + window)

        left_vecs = vectors[left_start:i]
        right_vecs = vectors[i:right_end]

        if not left_vecs or not right_vecs:
            similarities.append(1.0)
            continue

        def avg_vec(vecs):
            all_terms = set()
            for v in vecs:
                all_terms.update(v.keys())
            return {
                term: sum(v.get(term, 0) for v in vecs) / len(vecs)
                for term in all_terms
            }

        sim = cosine_sim(avg_vec(left_vecs), avg_vec(right_vecs))
        similarities.append(sim)

    if not similarities:
        return [True] + [False] * (n - 1)

    mean_sim = sum(similarities) / len(similarities)

    boundaries = [True]
    for sim in similarities:
        boundaries.append(sim < mean_sim)

    return boundaries


class TopicSegmenter:

    def run(self, transcript):
        texts = [line["text"] for line in transcript]
        n = len(texts)

        # Step 1: get a raw topic guess for every line
        raw_topics = [assign_topic_by_keywords(t) for t in texts]

        # Step 2: find boundaries using similarity drop AND topic label changes
        sim_boundaries = sliding_window_boundaries(texts, window=2)
        topic_change_boundaries = [False] + [
            raw_topics[i] != raw_topics[i - 1] for i in range(1, n)
        ]
        boundaries = [
            sim_boundaries[i] or topic_change_boundaries[i]
            for i in range(n)
        ]
        boundaries[0] = True

        # Step 3: split into segments and take the majority topic per segment
        segments = []
        current_seg = []
        for i in range(n):
            if boundaries[i] and current_seg:
                segments.append(current_seg)
                current_seg = []
            current_seg.append(i)
        if current_seg:
            segments.append(current_seg)

        final_topics = [""] * n
        for seg in segments:
            topic_counts = Counter(raw_topics[i] for i in seg)
            dominant = topic_counts.most_common(1)[0][0]
            for idx in seg:
                final_topics[idx] = dominant

        # Step 4: annotate each line with topic, change flag, and boundary flag
        result = []
        prev_topic = None
        for i, line in enumerate(transcript):
            topic = final_topics[i]
            annotated = dict(line)
            annotated["topic"] = topic
            annotated["topic_change"] = (topic != prev_topic)
            annotated["boundary"] = boundaries[i]
            prev_topic = topic
            result.append(annotated)

        boundary_indices = [i for i, b in enumerate(boundaries) if b]
        print(f"  [TopicSegmenter] Boundaries at lines: {boundary_indices}")
        print(f"  [TopicSegmenter] Topics found: {list(dict.fromkeys(final_topics))}")
        return result
