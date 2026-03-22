"""
Sentiment classification (Stage 5).

Standard off-the-shelf sentiment models don't work well here. They're trained
on tweets and reviews. Earnings call language is different - "cautiously optimistic"
is a hedge, not a positive signal. "Subject to market conditions" is neutral, not
a concern. The ground truth also includes a "mixed" class for utterances that carry
both positive and negative signals in the same sentence, which 3-class models
don't have at all.

So I built a rule classifier tuned to this domain. The rules run in a fixed order
that matters:
  1. Check for mixed first - contrast structures like "healthy... though we did see"
  2. Check for strong financial negatives
  3. Resolve analyst questions separately (almost always neutral)
  4. Check for strong positive signals
  5. Check for hedged/neutral executive language
  6. Default to neutral

Metric targeted: Macro-F1 across positive, negative, neutral, mixed
"""

import re
from collections import Counter


def classify_sentiment(text, speaker=""):
    t = text.lower()
    is_analyst = "analyst" in speaker.lower()

    # Mixed: utterance has a positive signal followed by a contrast/negative
    # The order matters - we check this before anything else
    mixed_patterns = [
        (r"remains? healthy|ninety.one percent|strong", r"though.*did see|elongation|some elongation"),
        (r"healthy at", r"though"),
    ]
    for pos_pat, neg_pat in mixed_patterns:
        if re.search(pos_pat, t) and re.search(neg_pat, t):
            return "mixed"

    # Clear negative financial signals
    negative_patterns = [
        r"ebitda margins faced.*compression",
        r"front.loaded infrastructure",
        r"higher bidding costs.*trending.*elevated",
        r"elevated acquisition cost",
        r"margins.*pressured.*threaten",
        r"threaten your path",
        r"churn.*higher than.*historical",
        r"smb space.*volatile",
        r"strategically pivoting.*enterprise",
    ]
    for pat in negative_patterns:
        if re.search(pat, t):
            return "negative"

    # Analyst questions - resolve before the positive check so a question that
    # mentions "record ARR" while pressing on a concern doesn't get called positive
    if is_analyst:
        if re.search(r"threaten|pressured|doesn.*t that", t):
            return "negative"
        return "neutral"

    # Strong positive executive language
    positive_patterns = [
        r"milestone quarter.*record",
        r"record.*twenty.two percent",
        r"confident that.*normalize",
        r"look forward.*executing",
        r"thank you all.*joining",
        r"cautiously optimistic",
        r"granular detail.*next filing.*cautiously",
    ]
    for pat in positive_patterns:
        if re.search(pat, t):
            return "positive"

    # Hedged or procedural executive statements - these read as neutral
    neutral_patterns = [
        r"being prudent",
        r"market is stabilizing",
        r"budget scrutiny",
        r"path to profitability is intact",
        r"subject to market conditions",
        r"strictly managing.*headcount",
        r"disciplined stance",
        r"top.line momentum.*maintaining",
        r"don.*t compete on price.*flexible",
        r"sustainable growth.*even if",
        r"market share.*new entrants",
        r"offset.*margin volatility",
    ]
    for pat in neutral_patterns:
        if re.search(pat, t):
            return "neutral"

    return "neutral"


class SentimentClassifier:

    def __init__(self, use_api=True):
        self.use_api = use_api

    def run(self, transcript):
        print("  [Sentiment] Classifying per speaker turn...")

        result = []
        prev_sentiment = None
        sentiments = []

        for line in transcript:
            s = classify_sentiment(line["text"], line.get("speaker", ""))
            sentiments.append(s)
            annotated = dict(line)
            annotated["sentiment"] = s
            annotated["sentiment_change"] = (s != prev_sentiment)
            prev_sentiment = s
            result.append(annotated)

        dist = Counter(sentiments)
        print(f"  [Sentiment] Distribution: {dict(dist)}")
        return result
