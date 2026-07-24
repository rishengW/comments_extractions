# Overseas China-travel mixed-method pipeline

This project turns the research-method specification in `questions.jpg` into a reproducible analysis stage. The image calls for a three-part design around the journey from **cognitive arousal -> trust building -> transaction fulfilment -> value spillover**:

1. **Big-data text mining** for TikTok and Trip.com public/exported data.
2. **Task-based usability testing** for observing concrete travel tasks.
3. **Semi-structured depth interviews** for explaining mechanisms behind observed behavior.

The primary code here implements the first stage end to end and produces the evidence needed to select texts for NVivo. The latter two stages are represented in the study scope below and should consume the same journey framework and topic/sentiment findings.

## Scope from the image

### Data collection

- TikTok video metadata under tags such as `#ChinaTravel` and `#ExploreChina`: video ID, publish time, likes, shares, and views/plays.
- TikTok comments: comment text, comment likes, commenter language or registration preference when lawfully available.
- Trip.com product/review data: product, rating, star distribution, review text, and traveller type (solo/family/business).
- Overseas Trip.com App feedback from Google Play and App Store.

Collection is intentionally an adapter boundary. Put lawful public exports or official-API results in `data/raw/`; do not bypass authentication, rate limits, robots rules, or platform terms. The pipeline does not store usernames or other direct identifiers. The clarified research question, inclusion rules, and keyword justification are in `RESEARCH_SCOPE.md`.

The included App Store collector can fetch visible Trip.com reviews directly:

```powershell
python -m src.collectors --country us --app-id 681752345 --max-reviews 20
```

This writes `data/raw/trip_app_store_reviews.jsonl` when explicitly run. The checked-in App Store pilot is kept under `data/optional/`; app-store reviews are not treated as China-tour product reviews.

Use the Selenium collector separately for each platform.

### Extract TikTok comments only

```powershell
python -m src.selenium_collectors `
  --source tiktok `
  --target-per-source 5000 `
  --max-videos-per-tag 50 `
  --max-comments-per-video 50 `
  --headed
```

TikTok comments are written to `data/raw/tiktok_comments_aggregate.jsonl`.

### Extract Trip.com product reviews only

```powershell
python -m src.selenium_collectors `
  --source trip `
  --target-per-source 5000 `
  --max-products-per-category 60 `
  --max-reviews-per-product 50 `
  --headed
```

Trip.com reviews are written to `data/raw/trip_product_reviews_aggregate.jsonl`.

`--source` accepts `all` (the default), `tiktok`, or `trip`. `--target-per-source` is the desired cumulative total for the selected source, including records already saved in the aggregate output. `--max-comments-per-video` and `--max-reviews-per-product` cap how many records can come from one video or product. Supply additional targets as `--tiktok-target "#ChinaTravel=https://..."` or `--trip-target "Shanghai tour=https://..."`.

Trip-only runs automatically use `data/selenium_trip_profile/`, separate from TikTok's persistent browser profile. Override it with `--profile-dir PATH` when needed.

Headed collection detects visible TikTok CAPTCHA or verification pages and waits up to 180 seconds for you to complete them manually in the Selenium browser. It never solves or bypasses the challenge. The cleared browser session is retained in `data/selenium_profile/` for later runs; keep that directory private. Change the manual window with `--verification-wait-seconds SECONDS`. Headless collection still stops immediately at a verification barrier. Usernames remain omitted from collected data.

TikTok comment collection is text-only. Images inside comments are not downloaded or extracted; text accompanying an image is retained, while image-only comments are skipped.

The collector automatically uses a project-local driver at `driver/chrome-win64/chromedriver-win64/chromedriver.exe` when present and uses the system Chrome browser. When Selenium Manager cannot download a compatible driver, select a local Chrome-for-Testing pair explicitly with `--chrome-binary PATH` and `--chromedriver-path PATH`.

### Text-mining procedure

1. Normalize exports to a common schema.
2. Remove duplicates, advertisements, emoji-only records, obvious bot/repeated content, and records shorter than the configured threshold. Keep English and the configured mainstream overseas-language allowlist. Tokenize with NLTK and remove language-matched NLTK stopwords where available, persisting the standardized result as `analysis_text`.
3. Fit LDA topic modeling to identify discussion hotspots such as visa policy, cross-border payment, cross-cultural attractions, and hotel service.
4. Score positive/negative/neutral sentiment with VADER when installed, with a deterministic lexicon fallback.
5. Proportionally sample 200–300 records by topic (default 250) for NVivo open coding and axial coding.

## Input contract

CSV, JSON, or JSONL files in `data/raw/` are accepted. The loader maps common aliases to these fields:

`record_id`, `source`, `content`, `published_at`, `likes`, `shares`, `views`, `rating`, `travel_type`, `language`, `hashtag`, `discovery_term`, `video_id`, `product_id`, `product_title`, `source_url`, `retrieved_at`.

`content` is the only required semantic field. File names are used as a source fallback, so names such as `tiktok_comments.csv` and `trip_reviews.jsonl` are useful.

## Run

```powershell
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m src.research_pipeline --input-dir data/raw --output-dir outputs --sample-size 250 --topics 8
```

Generated artifacts:

- `outputs/cleaned_records.csv` — de-duplicated, cleaned records and provenance.
- `outputs/analyzed_records.csv` — topic assignment/probability and sentiment label/score.
- `outputs/topics.csv` — top terms for every LDA topic.
- `outputs/nvivo_sample.csv` — proportional, deterministic sample for NVivo import.
- `outputs/run_summary.json` — counts and sentiment/source distribution for auditability.

The expanded collection findings and limitations are summarized in `reports/pilot_analysis.md`.

## Companion-method scope

The image only shows the start of the task-testing section, so the implementation does not invent task scripts. A follow-on study package should define observable tasks (discover a destination, evaluate trust, complete booking/payment, and post-trip sharing), capture completion time/errors/assistance/satisfaction, then use the topic and sentiment outputs to purposively recruit interviewees. Interviews should use a semi-structured guide mapped to the four journey phases and be coded with open and axial coding in NVivo. These additions are human-subject research and require consent, privacy review, and an approved data-management plan.
