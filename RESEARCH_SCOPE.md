# Clarified research query and justification

## Research question

How do overseas travellers describe the discovery, trust, transaction fulfilment, and experienced value of China travel products across short-video discussions and booking-platform reviews?

## Why these sources and terms are appropriate

TikTok comments capture the discovery and expectation stage. `#ChinaTravel` and `#ExploreChina` are the required entry points; `#TravelChina` is included because it is a direct semantic equivalent observed on the `#ExploreChina` result page. A comment is eligible because it appears under a post discovered through one of those tags, not because the comment itself repeats the tag.

Trip.com product reviews capture evaluation after a traveller has considered or used a bookable service. Eligible products must match `Shanghai Pass`, `China tour`, `Shanghai tour`, `Shanghai guide`, `city pass`, or named Shanghai tour attractions such as Yu Garden, the Bund, Huangpu River, or Sihang Warehouse. App-store reviews are not a substitute for these product reviews.

Together, the two sources cover different stages of the proposed framework:

- TikTok: cognitive arousal, destination image, and early trust signals.
- Trip.com: product trust, booking/service fulfilment, and post-use value.

## Inclusion rules

1. The page is publicly visible without authentication or bypassing a technical control.
2. TikTok records have a source-video URL, video ID, discovery tag, post hashtags, comment text, displayed date, and like count when visible.
3. Trip.com records have a product URL, product ID, product title, discovery keyword, review text, displayed date, and rating when visible.
4. Usernames, profile links, avatars, and developer/supplier responses are excluded from the research export.
5. Duplicate, emoji-only, bot-like, and non-target-language records are removed by the analysis pipeline rather than silently removed from the raw capture.

## Expanded collection completed on 21 July 2026

| Source query | Public records captured | Notes |
| --- | ---: | --- |
| TikTok `#ChinaTravel` | 195 | Comment records from 14 public tagged videos |
| TikTok `#ExploreChina` | 6 | Comment records from one video found on the tag page |
| TikTok `#TravelChina` | 4 | Related tag observed on the `#ExploreChina` page |
| Trip.com Shanghai tour/day-tour/guide products | 208 | Reviews from eight relevant Shanghai products |
| Trip.com `Shanghai Pass` | 0 | Indexed listings redirected to a live catalogue that exposed no pass-review text in the selected locale |

The canonical aggregate contains 413 raw records. The analysis pipeline retains 341 texts after NLTK tokenization and stopword cleaning plus de-duplication, advertisement, emoji/symbol-only, minimum-length, bot/noise, and language filtering, then draws the planned proportional 250-record NVivo sample. This remains a convenience sample of publicly visible comments and reviews rather than a probability sample; the collection manifest retains the zero-result `Shanghai Pass` query to make that coverage gap explicit.
