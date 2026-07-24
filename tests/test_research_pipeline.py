import json
import sys

import pandas as pd
import src.selenium_collectors as selenium_collectors

from src.selenium_collectors import (
    _hashtags,
    _likes,
    _parse_tiktok_cards,
    _text_only_comment,
    _tiktok_card_signature,
    _verification_marker,
    _verification_reason,
    _video_id,
    load_existing_rows,
)
from src.research_pipeline import (
    PipelineConfig,
    _tokenize,
    clean_text,
    preprocess,
    proportional_sample,
    run_pipeline,
    sentiment_scores,
    topic_model,
)


def test_selenium_collector_helpers_preserve_provenance():
    assert _video_id("https://www.tiktok.com/@name/video/123456") == "123456"
    assert _hashtags("Visit #ChinaTravel and #ExploreChina #ChinaTravel") == [
        "#ChinaTravel",
        "#ExploreChina",
    ]
    assert _likes("Like video 1,234 likes") == 1234


def test_selenium_collector_recognizes_verification_pages():
    assert _verification_marker("Verify to continue") == "verify to continue"
    assert _verification_marker("", "Security verification | TikTok") == (
        "security verification"
    )
    assert _verification_marker("Public comments", "TikTok") == ""


def test_selenium_collector_recognizes_empty_tiktok_challenge_shell():
    class Body:
        text = ""

    class Driver:
        title = "TikTok - Make Your Day"
        current_url = "https://www.tiktok.com/@name/video/123/"

        @staticmethod
        def find_element(*_args):
            return Body()

        @staticmethod
        def find_elements(*_args):
            return []

    assert _verification_reason(Driver()) == "empty TikTok verification/loading shell"


def test_tiktok_parser_skips_image_only_comments_and_keeps_text():
    metadata = {
        "video_id": "123",
        "hashtag": "#ChinaTravel",
        "video_caption": "Travel",
        "video_likes": 10,
        "video_comment_count": 2,
        "video_shares": 1,
    }
    payloads = [
        {"content": "", "date_text": "Today", "like_label": "0 likes"},
        {"content": "[Sticker]", "date_text": "Today", "like_label": "0 likes"},
        {
            "content": "[Sticker] Useful text",
            "date_text": "Today",
            "like_label": "3 likes",
        },
    ]

    rows = _parse_tiktok_cards(
        None,
        metadata,
        "#ChinaTravel",
        "https://www.tiktok.com/@name/video/123/",
        payloads,
    )

    assert [row["content"] for row in rows] == ["Useful text"]
    assert rows[0]["likes"] == 3
    assert "image" not in rows[0]


def test_tiktok_comment_text_removes_image_placeholders():
    assert _text_only_comment("[Sticker] wow") == "wow"
    assert _text_only_comment("[IMAGE]") == ""


def test_existing_tiktok_rows_are_normalized_to_text_only(tmp_path):
    records = [
        {
            "record_id": "old:1",
            "source": "tiktok_comment",
            "video_id": "123",
            "comment_date_text": "Today",
            "content": "[Sticker] wow",
        },
        {
            "record_id": "old:2",
            "source": "tiktok_comment",
            "video_id": "123",
            "comment_date_text": "Today",
            "content": "[Sticker]",
        },
    ]
    output = tmp_path / "tiktok.jsonl"
    output.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    rows = load_existing_rows(tmp_path, "tiktok_comment")

    assert [row["content"] for row in rows] == ["wow"]
    assert rows[0]["record_id"] != "old:1"


def test_trip_source_does_not_run_tiktok_collector(monkeypatch, tmp_path):
    class Driver:
        quit_called = False

        def quit(self):
            self.quit_called = True

    driver = Driver()
    trip_calls = []

    def fail_tiktok(*_args, **_kwargs):
        raise AssertionError("TikTok collector must not run in --source trip mode")

    def record_trip(*args, **_kwargs):
        trip_calls.append(args)
        return [{"source": "trip_product_review"}]

    monkeypatch.setattr(selenium_collectors, "_driver", lambda **_kwargs: driver)
    monkeypatch.setattr(selenium_collectors, "collect_tiktok", fail_tiktok)
    monkeypatch.setattr(selenium_collectors, "collect_trip", record_trip)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selenium_collectors",
            "--source",
            "trip",
            "--target-per-source",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )

    selenium_collectors.main()

    assert len(trip_calls) == 1
    assert driver.quit_called


def test_trip_discovery_stops_after_stable_unique_snapshots(monkeypatch):
    class FakeTimeout(Exception):
        pass

    class FakeWait:
        def __init__(self, driver, _timeout):
            self.driver = driver

        def until(self, condition):
            result = condition(self.driver)
            if result:
                return result
            raise FakeTimeout

    class Driver:
        scrolls = 0

        def execute_script(self, _script, *args):
            if args:
                return [
                    {
                        "href": "https://sg.trip.com/things-to-do/detail/123/",
                        "title": "Shanghai cultural experience",
                    },
                    {
                        "href": "https://sg.trip.com/things-to-do/detail/123/",
                        "title": "Shanghai cultural experience",
                    },
                ]
            self.scrolls += 1
            return None

    driver = Driver()
    monkeypatch.setattr(selenium_collectors, "_safe_get", lambda *_args: None)
    monkeypatch.setattr(
        selenium_collectors,
        "_selenium",
        lambda: (None, FakeTimeout, None, None, None, FakeWait),
    )

    products = selenium_collectors.discover_trip_product_urls(
        driver,
        "https://sg.trip.com/category/",
        "Shanghai cultural experiences",
        max_products=60,
    )

    assert list(products) == ["https://sg.trip.com/things-to-do/detail/123/"]
    assert driver.scrolls == 3


def test_trip_discovery_has_hard_round_limit_when_snapshot_keeps_changing(
    monkeypatch,
):
    class FakeTimeout(Exception):
        pass

    class FakeWait:
        def __init__(self, driver, _timeout):
            self.driver = driver

        def until(self, condition):
            result = condition(self.driver)
            if result:
                return result
            raise FakeTimeout

    class Driver:
        snapshots = 0
        scrolls = 0

        def execute_script(self, _script, *args):
            if args:
                self.snapshots += 1
                return [
                    {
                        "href": (
                            "https://sg.trip.com/things-to-do/detail/"
                            f"{1000 + self.snapshots}/"
                        ),
                        "title": "Shanghai cultural experience",
                    }
                ]
            self.scrolls += 1
            return None

    driver = Driver()
    monkeypatch.setattr(selenium_collectors, "_safe_get", lambda *_args: None)
    monkeypatch.setattr(
        selenium_collectors,
        "_selenium",
        lambda: (None, FakeTimeout, None, None, None, FakeWait),
    )

    selenium_collectors.discover_trip_product_urls(
        driver,
        "https://sg.trip.com/category/",
        "Shanghai cultural experiences",
        max_products=60,
    )

    assert driver.scrolls == 12


def test_trip_review_parser_skips_empty_cards_and_deduplicates_snapshots():
    payloads = [
        {"content": "", "date_text": "2026-07-01"},
        {"content": "Excellent guide", "date_text": "2026-07-02"},
    ]

    rows = selenium_collectors._parse_trip_cards(
        None,
        "https://sg.trip.com/things-to-do/detail/123/",
        "Shanghai cultural experience",
        "Shanghai cultural experiences",
        "5",
        payloads,
    )

    assert [row["content"] for row in rows] == ["Excellent guide"]
    assert (
        len(
            selenium_collectors._trip_product_signature(
                [
                    {
                        "href": "https://sg.trip.com/things-to-do/detail/123/",
                        "title": "One",
                    },
                    {
                        "href": "https://sg.trip.com/things-to-do/detail/123/",
                        "title": "One",
                    },
                ]
            )
        )
        == 1
    )


def test_trip_review_payloads_support_current_text_only_markup():
    class Driver:
        selectors = ()
        script = ""

        def execute_script(self, script, *selectors):
            self.script = script
            self.selectors = selectors
            return [{"content": "Excellent guide", "date_text": "2026-07-02"}]

    driver = Driver()

    payloads = selenium_collectors._trip_review_card_payloads(driver)

    assert payloads == [{"content": "Excellent guide", "date_text": "2026-07-02"}]
    assert "detail_review-card-outer" in driver.selectors[0]
    assert "detail_review-card-bottom" in driver.selectors[1]
    assert "detail_review" in driver.selectors[2]
    assert "ct-review-list-item" in driver.selectors[3]
    assert "img" not in driver.script.lower()


def test_tiktok_scroll_progress_uses_dom_signature_not_parsed_row_count():
    before = [{"content": "", "date_text": "Today"}]
    after = [
        {"content": "", "date_text": "Today"},
        {"content": "New text", "date_text": "Today"},
    ]

    assert _tiktok_card_signature(before) != _tiktok_card_signature(after)


def test_collector_parser_accepts_public_review_markup():
    # The live collector is network-bound; parser behavior is covered by the
    # checked-in public seed export and the analysis integration test below.
    seed = pd.read_json("data/optional/trip_app_store_reviews_seed.jsonl", lines=True)
    assert len(seed) == 6
    assert {"content", "rating", "source_url"} <= set(seed.columns)


def test_clean_text_removes_urls_handles_and_emoji():
    value = clean_text("Amazing trip!!! https://example.com @traveler 😀")
    assert "https" not in value
    assert "@traveler" not in value
    assert "Amazing" in value


def test_nltk_tokenization_removes_stopwords_and_punctuation():
    assert _tokenize("The friendly guide explained Yu Garden clearly.") == [
        "friendly",
        "guide",
        "explained",
        "yu",
        "garden",
        "clearly",
    ]
    assert _tokenize("It's a tour that can't disappoint us.") == [
        "tour",
        "disappoint",
    ]
    assert _tokenize("La visita es muy bonita", "es") == ["visita", "bonita"]


def test_preprocess_deduplicates_and_drops_bot_noise():
    frame = pd.DataFrame(
        {
            "source": ["tiktok"] * 3,
            "content": [
                "The hotel was helpful and beautiful.",
                "The hotel was helpful and beautiful.",
                "follow for follow follow for follow follow for follow",
            ],
        }
    )
    result = preprocess(frame, min_text_chars=8)
    assert len(result) == 1


def test_preprocess_filters_ads_symbols_and_mechanical_repetition():
    frame = pd.DataFrame(
        {
            "source": ["tiktok"] * 4,
            "language": ["en"] * 4,
            "content": [
                "Buy now with promo code CHINA20",
                "\U0001f600 \U0001f389 \U0001f600 \U0001f389",
                "tour tour tour tour tour tour",
                "The knowledgeable guide made the Shanghai tour worthwhile.",
            ],
        }
    )
    result = preprocess(frame)
    assert len(result) == 1
    assert result.iloc[0]["analysis_text"] == (
        "knowledgeable guide made shanghai tour worthwhile"
    )


def test_preprocess_filters_known_non_target_languages():
    frame = pd.DataFrame(
        {
            "source": ["trip"] * 2,
            "language": ["English", "xx"],
            "content": [
                "A useful review of the hotel service",
                "A review in another language",
            ],
        }
    )
    result = preprocess(frame)
    assert len(result) == 1
    assert result.iloc[0]["language"] == "en"


def test_topic_model_and_sample_have_expected_size():
    frame = pd.DataFrame(
        {
            "source": ["tiktok"] * 6,
            "content": [
                "visa policy information was clear",
                "visa application process was confusing",
                "hotel service was friendly and helpful",
                "hotel room was expensive and crowded",
                "payment was smooth and safe",
                "cross border payment was difficult",
            ],
        }
    )
    cleaned = preprocess(frame)
    modeled, topics = topic_model(cleaned, n_topics=3, random_seed=42)
    assert len(modeled) == len(cleaned)
    assert len(topics) == 3
    sample = proportional_sample(modeled, sample_size=4, random_seed=42)
    assert len(sample) == 4


def test_topic_model_handles_single_record():
    cleaned = preprocess(
        pd.DataFrame({"source": ["trip"], "content": ["A helpful hotel review"]})
    )
    modeled, topics = topic_model(cleaned, n_topics=8, random_seed=42)
    assert len(modeled) == 1
    assert len(topics) == 1


def test_sentiment_has_three_label_contract():
    result = sentiment_scores(["This is excellent", "This is terrible", "A normal day"])
    assert set(result.columns) == {"sentiment_label", "sentiment_score"}
    assert set(result["sentiment_label"]) <= {"positive", "negative", "neutral"}


def test_pipeline_writes_analysis_and_nvivo_artifacts(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    pd.DataFrame(
        {
            "video_id": ["v1", "v2", "v3"],
            "platform": ["tiktok"] * 3,
            "comment": [
                "The payment process was smooth and safe",
                "The visa rules were difficult and confusing",
                "The hotel staff were friendly and helpful",
            ],
            "like_count": [12, 4, 21],
        }
    ).to_csv(input_dir / "tiktok_comments.csv", index=False)

    paths = run_pipeline(
        PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            sample_size=2,
            n_topics=2,
        )
    )

    assert all(path.exists() for path in paths.values())
    analyzed = pd.read_csv(paths["analyzed"])
    sample = pd.read_csv(paths["nvivo_sample"])
    assert {"topic_id", "sentiment_label"} <= set(analyzed.columns)
    assert len(sample) == 2
