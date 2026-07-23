# China-travel comment-mining analysis

Collection date: 21 July 2026

## Corpus

- 205 public TikTok comment records from 16 videos: 195 discovered through `#ChinaTravel`, 6 through `#ExploreChina`, and 4 through the related `#TravelChina` tag.
- 208 public Trip.com product reviews from eight Shanghai tour, day-tour, attraction, and guide products.
- 413 records in the two canonical aggregate files. TikTok has 205 unique record IDs and 189 unique raw comment texts; Trip.com has 208 unique record IDs and review texts.
- 341 texts remained after NLTK tokenization and stopword cleaning plus de-duplication, advertisement, emoji/symbol-only, minimum-length, bot/noise, and language filtering: 134 TikTok comments and 207 Trip.com reviews.
- The pipeline produced the planned proportional 250-record NVivo sample.
- No usernames, profile links, avatars, or supplier responses are stored.

The live Trip.com catalogue exposed indexed `Shanghai Pass` listings but did not expose their review text in the selected locale. The zero-result query is retained in `data/collection_manifest.json`; unrelated App Store reviews were not substituted.

## Results

Sentiment among the 341 retained texts:

| Source | Positive | Neutral | Negative |
| --- | ---: | ---: | ---: |
| TikTok comments | 58 | 66 | 10 |
| Trip.com product reviews | 186 | 9 | 12 |
| **Total** | **244** | **75** | **22** |

The eight exploratory LDA topics consolidate into four useful qualitative areas:

1. Guide knowledge, communication, service, and recommendation intent.
2. Yu Garden, City God Temple, Shanghai landmarks, and attraction interpretation.
3. Tour itinerary, timing, transport, meeting points, and group logistics.
4. Destination beauty, emotional reactions, enjoyment, and perceived experience value.

TikTok comments are comparatively short and emphasize beauty, affect, and early destination interest. Trip.com reviews more often describe guide quality, service delivery, attraction interpretation, itinerary execution, and recommendation intent. This supports the intended distinction between discovery-stage discussion and fulfilment/post-use evaluation.

## Interpretation limits

- This is a convenience sample of publicly visible material, not a representative sample of all overseas China travellers.
- TikTok coverage is concentrated in 16 videos and is dominated by `#ChinaTravel`; comments qualify through the tagged source video and need not contain the tag themselves.
- Trip.com coverage is concentrated in Shanghai products, and 102 of 208 reviews come from one Yu Garden and City God Temple product.
- The selected Trip.com locale displays some reviews through platform translation. `language: en` records the visible English text, not a verified original language.
- Product reviews are strongly positive, so both source selection and visibility/order effects should be considered during qualitative coding.
- VADER is English-oriented and can misread irony, playful fear, short reactions, and translated phrasing. NVivo coding should resolve those cases manually.
- Page content and review ordering can change. The source URLs, product/video identifiers, and collection date are retained for auditability.
