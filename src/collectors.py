"""Lawful public-source collectors for the research pipeline.

The collectors use public pages or official endpoints and keep collection separate
from analysis. They do not log in, solve challenges, or attempt to bypass platform
controls. This module covers the optional App Store source; public TikTok and
Trip.com product-page collection is implemented in ``selenium_collectors``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


TRIP_APP_STORE_REVIEW_URL = (
    "https://apps.apple.com/{country}/app/{app_id}"
    "?see-all=reviews&platform=iphone"
)


def _record_id(app_id: str, title: str, published_at: str, content: str) -> str:
    digest = hashlib.sha1(
        f"{app_id}|{title}|{published_at}|{content}".encode("utf-8")
    ).hexdigest()[:16]
    return f"apple:{app_id}:{digest}"


def fetch_trip_app_store_reviews(
    app_id: str = "681752345",
    country: str = "us",
    max_reviews: int = 20,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch visible Trip.com App Store reviews without usernames or responses."""
    url = TRIP_APP_STORE_REVIEW_URL.format(country=country, app_id=app_id)
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (research export; contact project owner)"},
        timeout=timeout,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    rows: list[dict[str, object]] = []
    for heading in soup.find_all("h3"):
        card = heading
        for _ in range(8):
            if card is None:
                break
            time_node = card.find("time")
            review_paragraph = card.find("p")
            rating_node = next(
                (
                    node
                    for node in card.find_all(attrs={"aria-label": True})
                    if re.search(r"\d+ Stars?", node.get("aria-label", ""))
                ),
                None,
            )
            if time_node is not None and review_paragraph is not None and rating_node is not None:
                title = heading.get_text(" ", strip=True)
                content = review_paragraph.get_text(" ", strip=True)
                published_at = time_node.get("datetime", "")
                rating_match = re.search(r"\d+", rating_node.get("aria-label", ""))
                rating = int(rating_match.group(0)) if rating_match else None
                rows.append(
                    {
                        "record_id": _record_id(app_id, title, published_at, content),
                        "source": "trip_app_store",
                        "content": content,
                        "review_title": title,
                        "published_at": published_at,
                        "rating": rating,
                        "language": "en",
                        "app_id": app_id,
                        "source_url": url,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                break
            card = card.parent
        if len(rows) >= max_reviews:
            break
    if not rows:
        raise RuntimeError(
            "No review cards were found. The public page may have changed or blocked the request."
        )
    return pd.DataFrame(rows)


def write_jsonl(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-id", default="681752345")
    parser.add_argument("--country", default="us")
    parser.add_argument("--max-reviews", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/trip_app_store_reviews.jsonl"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = fetch_trip_app_store_reviews(
        app_id=args.app_id,
        country=args.country,
        max_reviews=args.max_reviews,
    )
    write_jsonl(frame, args.output)
    print(f"wrote {len(frame)} reviews to {args.output}")


if __name__ == "__main__":
    main()
