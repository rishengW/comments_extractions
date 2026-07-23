"""Reproducible text-mining pipeline for TikTok and Trip.com exports.

The pipeline deliberately consumes exported files rather than automating access to
platforms. Collection adapters can write CSV/JSON/JSONL files using the canonical
fields documented in README.md, then this module performs cleaning, topic modeling,
sentiment analysis, and proportional NVivo sampling.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


CANONICAL_COLUMNS = [
    "record_id",
    "source",
    "content",
    "review_title",
    "published_at",
    "likes",
    "shares",
    "views",
    "rating",
    "travel_type",
    "language",
    "hashtag",
    "discovery_term",
    "video_id",
    "video_caption",
    "comment_date_text",
    "product_id",
    "product_title",
    "app_id",
    "source_url",
    "retrieved_at",
]

LANGUAGE_ALLOWLIST = {
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "ja",
    "ko",
    "th",
    "vi",
    "id",
    "ms",
    "ar",
    "ru",
    "zh",
}

LANGUAGE_NAMES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "thai": "th",
    "vietnamese": "vi",
    "indonesian": "id",
    "malay": "ms",
    "arabic": "ar",
    "russian": "ru",
    "chinese": "zh",
}

POSITIVE_WORDS = {
    "amazing",
    "beautiful",
    "best",
    "enjoy",
    "excellent",
    "excited",
    "friendly",
    "good",
    "great",
    "happy",
    "helpful",
    "love",
    "loved",
    "perfect",
    "recommend",
    "safe",
    "smooth",
    "wonderful",
}

NEGATIVE_WORDS = {
    "avoid",
    "bad",
    "confusing",
    "crowded",
    "disappointed",
    "difficult",
    "expensive",
    "hate",
    "long",
    "lost",
    "negative",
    "poor",
    "problem",
    "refund",
    "rude",
    "slow",
    "stressful",
    "terrible",
    "unhelpful",
    "wait",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)[@#][\w.-]+", re.UNICODE)
EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
TOKEN_RE = re.compile(r"(?u)^(?=.*[^\W\d_])[\w'-]{2,}$")

AD_PHRASES = {
    "affiliate link",
    "buy now",
    "check my bio",
    "click the link",
    "contact me on",
    "discount code",
    "dm me",
    "follow for follow",
    "free followers",
    "link in bio",
    "message me on",
    "promo code",
    "sponsored offer",
}

TOKENIZER_ARTIFACTS = {
    "'d",
    "'ll",
    "'m",
    "'re",
    "'s",
    "'ve",
    "ca",
    "n't",
    "us",
    "wo",
}

NLTK_STOPWORD_LANGUAGES = {
    "ar": "arabic",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "id": "indonesian",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "zh": "chinese",
}


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    sample_size: int = 250
    n_topics: int = 8
    random_seed: int = 42
    min_text_chars: int = 8


def _read_file(path: Path) -> pd.DataFrame:
    """Read one export while preserving all fields supplied by the collector."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("items", [payload]))
        return pd.json_normalize(payload)
    raise ValueError(f"Unsupported input file: {path.name}")


def load_exports(input_dir: Path) -> pd.DataFrame:
    """Load CSV, JSON, and JSONL exports from a directory."""
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl"}
    )
    if not files:
        raise FileNotFoundError(f"No CSV, JSON, or JSONL files found in {input_dir}")
    frames = []
    for path in files:
        frame = _read_file(path)
        frame["source"] = frame.get("source", path.stem.split("_")[0])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _first_present(row: pd.Series, names: Sequence[str], default=""):
    for name in names:
        if name in row.index and pd.notna(row[name]) and str(row[name]).strip():
            return row[name]
    return default


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common platform export names to a stable cross-source schema."""
    aliases = {
        "record_id": ["record_id", "id", "video_id", "review_id", "comment_id"],
        "content": ["content", "text", "comment", "review_text", "caption", "body"],
        "review_title": ["review_title", "title", "headline"],
        "published_at": ["published_at", "publish_time", "created_at", "date"],
        "likes": ["likes", "like_count", "comment_likes"],
        "shares": ["shares", "share_count"],
        "views": ["views", "play_count", "view_count", "plays"],
        "rating": ["rating", "stars", "score"],
        "travel_type": ["travel_type", "trip_type", "traveler_type"],
        "language": ["language", "lang", "commenter_language"],
        "hashtag": ["hashtag", "hashtags", "tag"],
        "discovery_term": ["discovery_term", "search_term", "query"],
        "video_id": ["video_id"],
        "video_caption": ["video_caption", "caption"],
        "comment_date_text": ["comment_date_text"],
        "product_id": ["product_id"],
        "product_title": ["product_title", "product_name"],
        "app_id": ["app_id", "application_id"],
        "source_url": ["source_url", "url", "review_url"],
        "retrieved_at": ["retrieved_at", "collected_at"],
        "source": ["source", "platform", "site"],
    }
    result = pd.DataFrame(index=frame.index)
    for canonical_name, names in aliases.items():
        result[canonical_name] = frame.apply(
            lambda row: _first_present(row, names), axis=1
        )
    result["record_id"] = result["record_id"].astype(str)
    missing_ids = result["record_id"].isin({"", "nan", "None"})
    result.loc[missing_ids, "record_id"] = [
        f"row-{index}" for index in result.index[missing_ids]
    ]
    result["source"] = result["source"].replace("", "unknown").astype(str).str.lower()
    result["content"] = result["content"].fillna("").astype(str)
    return result[CANONICAL_COLUMNS]


def detect_language(text: str) -> str:
    """Use langdetect when installed and return ``other`` for excluded languages."""
    letter_text = re.sub(r"[^A-Za-z]", "", text)
    if letter_text and len(letter_text) < 24 and all(ord(char) < 128 for char in text):
        # Statistical language detection is unreliable on very short reviews.
        return "en"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        language = detect(text)
        return language if language in LANGUAGE_ALLOWLIST else "other"
    except Exception:
        # A deterministic fallback avoids making the pipeline fail in minimal envs.
        return "en" if re.search(r"[A-Za-z]", text) else "other"


def normalize_language(value: object) -> str:
    """Normalize ISO codes and common language names from platform exports."""
    raw = str(value).strip().lower().replace("_", "-")
    if raw in {"", "nan", "none"}:
        return ""
    return LANGUAGE_NAMES.get(raw, raw.split("-")[0])


def clean_text(text: str) -> str:
    """Remove URLs, handles, emoji-only noise, and obvious bot repetition."""
    text = unicodedata.normalize("NFKC", str(text))
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = REPEATED_CHAR_RE.sub(r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=None)
def _language_stopwords(language: str) -> frozenset[str]:
    """Load the matching NLTK stopwords where that corpus is available."""
    corpus_language = NLTK_STOPWORD_LANGUAGES.get(language)
    if corpus_language is None:
        return frozenset(TOKENIZER_ARTIFACTS)
    try:
        return (
            frozenset(nltk_stopwords.words(corpus_language)) | TOKENIZER_ARTIFACTS
        )
    except LookupError as error:
        raise RuntimeError(
            "The NLTK stopwords corpus is required. Run "
            "`python -m nltk.downloader stopwords` before the pipeline."
        ) from error


def _word_tokens(text: str) -> list[str]:
    """Tokenize normalized text with NLTK without requiring Punkt sentence data."""
    normalized = unicodedata.normalize("NFKC", str(text))
    return [
        token.casefold()
        for token in word_tokenize(normalized, preserve_line=True)
        if TOKEN_RE.fullmatch(token)
    ]


def _tokenize(text: str, language: str = "en") -> list[str]:
    """Return NLTK tokens after language-aware stopword cleaning."""
    stopwords = _language_stopwords(language)
    return [token for token in _word_tokens(text) if token not in stopwords]


def _looks_like_ad(text: str) -> bool:
    lowered = text.casefold()
    return any(phrase in lowered for phrase in AD_PHRASES)


def _looks_like_bot(text: str) -> bool:
    words = _word_tokens(text)
    if len(words) < 3:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.35


def preprocess(frame: pd.DataFrame, min_text_chars: int = 8) -> pd.DataFrame:
    """Apply the study's NLTK normalization and required noise filters."""
    result = canonicalize(frame)
    result["clean_text"] = result["content"].map(clean_text)
    result["language"] = result["language"].map(normalize_language)
    missing_language = result["language"].eq("")
    result.loc[missing_language, "language"] = result.loc[
        missing_language, "clean_text"
    ].map(detect_language)
    result["analysis_text"] = result.apply(
        lambda row: " ".join(_tokenize(row["clean_text"], row["language"])), axis=1
    )
    dedupe_key = result["clean_text"].str.casefold()
    keep = (
        result["clean_text"].str.len().ge(min_text_chars)
        & result["clean_text"].map(lambda text: any(char.isalnum() for char in text))
        & ~result["clean_text"].map(_looks_like_ad)
        & ~result["clean_text"].map(_looks_like_bot)
        & ~dedupe_key.duplicated()
        & result["language"].isin(LANGUAGE_ALLOWLIST)
        & result["analysis_text"].ne("")
    )
    return result.loc[keep].reset_index(drop=True)


def topic_model(frame: pd.DataFrame, n_topics: int, random_seed: int):
    """Fit an LDA model and return document assignments and topic terms."""
    if frame.empty:
        return frame.assign(topic_id=pd.Series(dtype="int64"), topic_probability=pd.Series(dtype=float)), pd.DataFrame(columns=["topic_id", "top_terms"])
    documents = (
        frame["analysis_text"]
        if "analysis_text" in frame
        else frame["clean_text"].map(lambda text: " ".join(_tokenize(text)))
    )
    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=1 if len(frame) < 20 else 2,
        max_df=1.0 if len(frame) < 20 else 0.98,
    )
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError as error:
        if "empty vocabulary" not in str(error).lower():
            raise
        # A highly uniform export can still have no terms after max/min DF.
        vectorizer = CountVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            min_df=1,
            max_df=1.0,
        )
        matrix = vectorizer.fit_transform(documents)
    n_topics = max(1, min(n_topics, matrix.shape[0], matrix.shape[1]))
    model = LatentDirichletAllocation(
        n_components=n_topics, random_state=random_seed, learning_method="batch"
    )
    distribution = model.fit_transform(matrix)
    terms = vectorizer.get_feature_names_out()
    topic_rows = []
    for topic_id, weights in enumerate(model.components_):
        top_indices = weights.argsort()[::-1][:10]
        topic_rows.append(
            {"topic_id": topic_id, "top_terms": ", ".join(terms[top_indices])}
        )
    assigned = frame.copy()
    assigned["topic_id"] = distribution.argmax(axis=1).astype(int)
    assigned["topic_probability"] = distribution.max(axis=1).round(6)
    return assigned, pd.DataFrame(topic_rows)


def _lexicon_sentiment(text: str) -> tuple[str, float]:
    tokens = set(_tokenize(text))
    positive = len(tokens & POSITIVE_WORDS)
    negative = len(tokens & NEGATIVE_WORDS)
    raw = positive - negative
    if raw == 0:
        return "neutral", 0.0
    score = min(1.0, abs(raw) / max(3, len(tokens)))
    return ("positive" if raw > 0 else "negative"), round(score, 6)


def sentiment_scores(texts: Iterable[str]) -> pd.DataFrame:
    """Score text using VADER when available, otherwise a deterministic lexicon."""
    texts = list(texts)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        rows = []
        for text in texts:
            compound = analyzer.polarity_scores(text)["compound"]
            label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
            rows.append({"sentiment_label": label, "sentiment_score": round(compound, 6)})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(
            [_lexicon_sentiment(text) for text in texts],
            columns=["sentiment_label", "sentiment_score"],
        )


def proportional_sample(
    frame: pd.DataFrame, sample_size: int, random_seed: int, strata: str = "topic_id"
) -> pd.DataFrame:
    """Sample proportionally by LDA topic, with deterministic largest-remainder quotas."""
    if frame.empty:
        return frame.copy()
    target = min(max(1, sample_size), len(frame))
    counts = frame[strata].value_counts().sort_index()
    raw = counts / len(frame) * target
    quotas = raw.map(math.floor).astype(int)
    remainder = target - int(quotas.sum())
    if remainder:
        order = (raw - quotas).sort_values(ascending=False).index.tolist()
        for key in order[:remainder]:
            quotas.loc[key] += 1
    pieces = []
    for key, quota in quotas.items():
        group = frame[frame[strata] == key]
        if quota:
            pieces.append(group.sample(n=min(int(quota), len(group)), random_state=random_seed))
    result = pd.concat(pieces, ignore_index=True) if pieces else frame.head(0).copy()
    return result.sample(frac=1, random_state=random_seed).reset_index(drop=True)


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    """Run all stages and write analysis-ready CSV artifacts."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_exports(config.input_dir)
    cleaned = preprocess(raw, config.min_text_chars)
    modeled, topics = topic_model(cleaned, config.n_topics, config.random_seed)
    sentiments = sentiment_scores(modeled["clean_text"])
    analyzed = pd.concat([modeled.reset_index(drop=True), sentiments], axis=1)
    sample = proportional_sample(analyzed, config.sample_size, config.random_seed)

    paths = {
        "cleaned": config.output_dir / "cleaned_records.csv",
        "topics": config.output_dir / "topics.csv",
        "analyzed": config.output_dir / "analyzed_records.csv",
        "nvivo_sample": config.output_dir / "nvivo_sample.csv",
        "summary": config.output_dir / "run_summary.json",
    }
    cleaned.to_csv(paths["cleaned"], index=False, encoding="utf-8-sig")
    topics.to_csv(paths["topics"], index=False, encoding="utf-8-sig")
    analyzed.to_csv(paths["analyzed"], index=False, encoding="utf-8-sig")
    sample.to_csv(paths["nvivo_sample"], index=False, encoding="utf-8-sig")
    summary = {
        "input_records": int(len(raw)),
        "retained_records": int(len(analyzed)),
        "topic_count": int(len(topics)),
        "nvivo_sample_records": int(len(sample)),
        "sources": analyzed["source"].value_counts().to_dict() if not analyzed.empty else {},
        "sentiment": analyzed["sentiment_label"].value_counts().to_dict() if not analyzed.empty else {},
        "preprocessing": {
            "tokenizer": "nltk.word_tokenize(preserve_line=True)",
            "stopwords": "language-matched nltk.corpus.stopwords where available",
            "filters": [
                "duplicate",
                "advertisement",
                "emoji-or-symbol-only",
                "mechanically-repeated-or-bot-like",
                "below-minimum-length",
                "non-target-language",
            ],
        },
        "topic_model": "scikit-learn LatentDirichletAllocation",
        "sentiment_model": "VADER with deterministic lexicon fallback",
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--topics", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-text-chars", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = run_pipeline(
        PipelineConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            n_topics=args.topics,
            random_seed=args.seed,
            min_text_chars=args.min_text_chars,
        )
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
