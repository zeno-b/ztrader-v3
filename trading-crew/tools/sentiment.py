"""Sentiment utility helpers for research agent preprocessing."""

from __future__ import annotations


def clamp_sentiment(score: float) -> float:
    """Clamp sentiment score to [-1.0, 1.0]."""

    return max(-1.0, min(1.0, score))
