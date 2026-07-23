"""Resumable Selenium collection for public TikTok and Trip.com comments.

The collector stops at authentication or verification barriers. It does not evade
rate limits, CAPTCHAs, login requirements, or platform controls. Public usernames,
profile URLs, and avatars are deliberately excluded from the research export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


DEFAULT_TIKTOK_TARGETS = {
    "#ChinaTravel": "https://www.tiktok.com/@pumpkin_uncle/video/7602417845045136695",
    "#ExploreChina": "https://www.tiktok.com/@explorechina786/video/7596389636167978261",
    "#TravelChina": "https://www.tiktok.com/@nullwhisper0/video/7643621133257870606",
}

DEFAULT_TIKTOK_TAGS = (
    "#ChinaTravel",
    "#ExploreChina",
    "#TravelChina",
    "#VisitChina",
    "#ChinaTour",
)

DEFAULT_TRIP_TARGETS = {
    "Shanghai day tour": "https://sg.trip.com/things-to-do/detail/88901396/",
}

DEFAULT_TRIP_CATEGORIES = {
    "Shanghai day tour": "https://sg.trip.com/things-to-do/experiences/shanghai-day-tour/",
    "Shanghai cultural experiences": "https://sg.trip.com/things-to-do/experiences/shanghai-culturalexperiences/",
    "Shanghai city sightseeing": "https://sg.trip.com/things-to-do/experiences/shanghai-citysightseeing/",
    "Shanghai city pass": "https://sg.trip.com/things-to-do/experiences/shanghai-city-pass/",
}

TRIP_RELEVANCE_TERMS = (
    "shanghai pass",
    "china tour",
    "shanghai tour",
    "shanghai guide",
    "city pass",
    "day tour",
    "cultural experience",
    "city sightseeing",
    "yu garden",
    "the bund",
    "huangpu river",
    "sihang warehouse",
    "zhujiajiao",
    "oriental pearl",
    "shanghai disney",
    "shanghai museum",
)

VERIFICATION_MARKERS = (
    "complete the captcha",
    "security verification",
    "verify to continue",
    "verify you are human",
    "verification required",
    "drag the puzzle piece",
    "slide to fit the puzzle",
    "perform the following verification",
)

VERIFICATION_SELECTORS = (
    'iframe[src*="captcha" i]',
    'iframe[src*="verify" i]',
    '[id*="captcha" i]',
    '[class*="captcha" i]',
)

TIKTOK_COMMENT_BUTTON_SELECTOR = (
    'button[aria-label^="Read or add comments"], '
    'button[data-e2e="comment-icon"], '
    '[role="button"][data-e2e="comment-icon"]'
)
TIKTOK_COMMENT_CARD_SELECTOR = (
    'div[class*="DivCommentItemWrapper"], div[data-e2e="comment-level-1"]'
)
TIKTOK_EMPTY_COMMENT_MARKERS = (
    "comments are turned off",
    "no comments yet",
    "be the first to comment",
)

TRIP_PRODUCT_LINK_SELECTOR = 'a[href*="/things-to-do/detail/"]'
TRIP_REVIEW_CARD_SELECTOR = (
    'div[class*="detail_review-card-outer"], '
    "div.ct-review-list-item, "
    'div[class*="review_section_impression_item__"], '
    'div[class*="review_item"], div[class*="reviewItem"], '
    'div[class*="comment_item"], div[class*="commentItem"]'
)
TRIP_REVIEW_CONTENT_SELECTOR = (
    '[data-testid="vac_comment_detail_main_content_text"], '
    '[class~="ct-review-evaluation-detail"], '
    '[class*="detail_review-card-bottom"], '
    '[class*="impression_item_content"], '
    '[class*="review_item_content"], [class*="comment_item_content"]'
)
TRIP_REVIEW_DATE_SELECTOR = (
    "time[datetime], [datetime], "
    '[data-testid="vac_comment_detail_tourtype_ipAttributionName"], '
    '[class~="ct-review-text-4"], '
    '[class*="detail_review-card-name-2"], '
    '[class*="detail_review-date"], [class*="detail_review-time"], '
    '[class*="review"][class*="date"], [class*="review"][class*="time"], '
    '[class*="impression_item_time"], [class*="review_item_time"], '
    '[class*="comment_item_time"]'
)
TRIP_REVIEW_SECTION_SELECTOR = 'div[class*="activity-review-section_review_section__"]'
TRIP_FULL_REVIEW_CONTROL_SELECTOR = 'div[class*="detail_review-desc-right"]'
TRIP_FULL_REVIEW_CARD_SELECTOR = "div.ct-review-list-item"
TRIP_NEXT_REVIEW_PAGE_SELECTOR = (
    ".ct-review-pagination-next:not(.ct-review-pagination-disabled)"
)


class VerificationBarrierError(RuntimeError):
    """Raised when collection cannot proceed without manual verification."""


class TikTokPageError(RuntimeError):
    """Raised when a loaded TikTok page does not expose the expected public UI."""


class TripPageError(RuntimeError):
    """Raised when a loaded Trip.com page does not expose the expected public UI."""


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _video_id(url: str) -> str:
    match = re.search(r"/video/(\d+)", url)
    return match.group(1) if match else ""


def _product_id(url: str) -> str:
    match = re.search(r"/detail/(\d+)", url)
    return match.group(1) if match else ""


def _hashtags(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"#[\w\u0080-\uffff]+", text)))


def _likes(label: str) -> int:
    match = re.search(r"([\d,.]+)\s+likes?", label, re.IGNORECASE)
    return int(match.group(1).replace(",", "")) if match else 0


def _text_only_comment(value: str) -> str:
    value = re.sub(r"\[(?:sticker|image|photo|gif)\]", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def _canonical_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path.rstrip("/") + "/", "", "")
    )


def _selenium():
    try:
        from selenium import webdriver
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Selenium is required for live collection. Install requirements.txt first."
        ) from error
    return webdriver, TimeoutException, Options, By, EC, WebDriverWait


def _driver(
    headless: bool,
    chrome_binary: Path | None = None,
    chromedriver_path: Path | None = None,
    profile_dir: Path | None = None,
):
    webdriver, _, Options, _, _, _ = _selenium()
    options = Options()
    options.page_load_strategy = "eager"
    if chrome_binary:
        options.binary_location = str(chrome_binary)
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-notifications")
    options.add_argument("--no-sandbox")
    options.add_argument("--remote-debugging-pipe")
    options.add_argument("--lang=en-GB")
    options.add_argument("--window-size=1440,1200")
    if profile_dir:
        profile_dir.mkdir(parents=True, exist_ok=True)
        options.add_argument(f"--user-data-dir={profile_dir.resolve()}")
    if chromedriver_path:
        from selenium.webdriver.chrome.service import Service

        driver = webdriver.Chrome(
            service=Service(executable_path=str(chromedriver_path)),
            options=options,
        )
    else:
        driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(35)
    driver.set_script_timeout(15)
    return driver


def _default_chromedriver_path() -> Path | None:
    """Find the project-local driver before falling back to Selenium Manager."""
    project_root = Path(__file__).resolve().parents[1]
    candidates = (
        project_root
        / "driver"
        / "chrome-win64"
        / "chromedriver-win64"
        / "chromedriver.exe",
        project_root / "driver" / "chromedriver-win64" / "chromedriver.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    discovered = shutil.which("chromedriver")
    return Path(discovered) if discovered else None


def _verification_marker(page_text: str, title: str = "", url: str = "") -> str:
    visible_page = " ".join((page_text, title, url)).lower()
    return next(
        (marker for marker in VERIFICATION_MARKERS if marker in visible_page), ""
    )


def _verification_reason(driver: Any) -> str:
    _, _, _, By, _, _ = _selenium()
    try:
        visible_text = driver.find_element(By.TAG_NAME, "body").text
    except Exception:
        visible_text = ""
    title = str(getattr(driver, "title", "") or "")
    url = str(getattr(driver, "current_url", "") or "")
    marker = _verification_marker(visible_text, title, url)
    if marker:
        return f'page text "{marker}"'
    for selector in VERIFICATION_SELECTORS:
        try:
            if any(
                element.is_displayed()
                for element in driver.find_elements(By.CSS_SELECTOR, selector)
            ):
                return f'visible element matching "{selector}"'
        except Exception:
            continue
    if (
        "tiktok.com" in url.lower()
        and not visible_text.strip()
        and title.strip().lower() == "tiktok - make your day"
    ):
        return "empty TikTok verification/loading shell"
    return ""


def _wait_for_manual_verification(driver: Any, wait_seconds: int) -> bool:
    """Wait for the user to clear a visible verification barrier in headed Chrome."""
    _, TimeoutException, _, _, _, WebDriverWait = _selenium()
    reason = _verification_reason(driver)
    if not reason:
        return False
    if reason == "empty TikTok verification/loading shell":
        try:
            WebDriverWait(driver, 5, poll_frequency=0.25).until(
                lambda current: not _verification_reason(current)
            )
        except TimeoutException:
            pass
        else:
            return False
    if wait_seconds <= 0:
        raise VerificationBarrierError(
            "TikTok verification detected. Rerun with --headed to complete it "
            "manually in the Selenium browser."
        )
    _log(
        f"TikTok: verification detected ({reason}). Complete it in the browser; "
        f"waiting up to {wait_seconds} seconds"
    )
    try:
        WebDriverWait(driver, wait_seconds, poll_frequency=1).until(
            lambda current: not _verification_reason(current)
        )
    except TimeoutException as error:
        raise VerificationBarrierError(
            f"TikTok verification was not cleared within {wait_seconds} seconds."
        ) from error
    _log("TikTok: verification cleared; resuming collection")
    return True


def _wait_for_tiktok_ui(
    driver: Any,
    condition: Any,
    stage: str,
    verification_wait_seconds: int,
    timeout_seconds: int = 20,
) -> Any:
    _, TimeoutException, _, _, _, WebDriverWait = _selenium()
    try:
        return WebDriverWait(driver, timeout_seconds).until(condition)
    except TimeoutException as error:
        cause = error
        if _wait_for_manual_verification(driver, verification_wait_seconds):
            try:
                return WebDriverWait(driver, timeout_seconds).until(condition)
            except TimeoutException as retry_error:
                cause = retry_error
        title = str(getattr(driver, "title", "") or "")[:120]
        url = str(getattr(driver, "current_url", "") or "")[:250]
        raise TikTokPageError(
            f"TikTok {stage} did not appear within {timeout_seconds} seconds "
            f'(title="{title}", url="{url}").'
        ) from cause


def _safe_get(driver: Any, url: str, verification_wait_seconds: int = 0) -> None:
    _, TimeoutException, _, By, EC, WebDriverWait = _selenium()
    try:
        driver.get(url)
    except TimeoutException:
        driver.execute_script("window.stop();")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    _wait_for_manual_verification(driver, verification_wait_seconds)


def _dedupe_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    result = {}
    for row in rows:
        key = str(row.get("record_id") or "")
        if not key:
            key = _stable_id(
                str(row.get("source") or "record"),
                str(row.get("source_url") or ""),
                str(row.get("content") or ""),
            )
            row = {**row, "record_id": key}
        result[key] = row
    return list(result.values())


def load_existing_rows(input_dir: Path, source: str) -> list[dict[str, object]]:
    rows = []
    for path in sorted(input_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == source:
                if source == "tiktok_comment":
                    content = _text_only_comment(str(row.get("content") or ""))
                    if not content:
                        continue
                    if content != row.get("content"):
                        row = {
                            **row,
                            "content": content,
                            "record_id": _stable_id(
                                "tiktok",
                                str(row.get("video_id") or ""),
                                str(row.get("comment_date_text") or ""),
                                content,
                            ),
                        }
                rows.append(row)
    return _dedupe_rows(rows)


def write_jsonl(rows: Iterable[dict[str, object]], output: Path) -> None:
    rows = _dedupe_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda value: str(value["record_id"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_error(error: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(error, ensure_ascii=False) + "\n")


def _log(message: str) -> None:
    print(message, flush=True)


def _brief_error(error: Exception) -> str:
    message = str(error).splitlines()[0].strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def discover_tiktok_video_urls(
    driver: Any,
    discovery_tag: str,
    max_videos: int = 12,
    verification_wait_seconds: int = 0,
) -> list[str]:
    """Discover public video URLs from one TikTok hashtag result page."""
    _, TimeoutException, _, By, EC, WebDriverWait = _selenium()
    slug = re.sub(r"[^\w]", "", discovery_tag).lower()
    _safe_get(
        driver,
        f"https://www.tiktok.com/tag/{slug}",
        verification_wait_seconds,
    )
    refresh = driver.find_elements(By.XPATH, '//button[normalize-space()="Refresh"]')
    if refresh:
        refresh[0].click()
    selector = 'a[href*="/video/"]'
    _wait_for_tiktok_ui(
        driver,
        EC.presence_of_element_located((By.CSS_SELECTOR, selector)),
        f"video links for {discovery_tag}",
        verification_wait_seconds,
    )

    urls: dict[str, None] = {}
    stable_rounds = 0
    while len(urls) < max_videos and stable_rounds < 3:
        previous = len(urls)
        for link in driver.find_elements(By.CSS_SELECTOR, selector):
            href = link.get_attribute("href") or ""
            if _video_id(href):
                urls[_canonical_url(href)] = None
                if len(urls) >= max_videos:
                    break
        if len(urls) >= max_videos:
            break
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            WebDriverWait(driver, 4).until(
                lambda current: (
                    len(current.find_elements(By.CSS_SELECTOR, selector)) > previous
                )
            )
        except TimeoutException:
            stable_rounds += 1
        else:
            stable_rounds = 0
    return list(urls)[:max_videos]


def _tiktok_metadata(driver: Any, video_url: str) -> dict[str, object]:
    snapshot = driver.execute_script(
        """
        const caption = [...document.querySelectorAll('article img[alt]')]
          .map((image) => image.getAttribute('alt') || '')
          .find((value) => value.includes('#')) || '';
        const labels = [...document.querySelectorAll('button[aria-label]')]
          .map((button) => button.getAttribute('aria-label') || '');
        return {caption, labels};
        """
    ) or {"caption": "", "labels": []}
    caption = str(snapshot.get("caption") or "")
    stats = {"likes": 0, "comment_count": 0, "shares": 0}
    for label in snapshot.get("labels") or []:
        match = re.search(r"([\d,.]+)\s+(likes|comments|shares)", label, re.IGNORECASE)
        if match:
            value = int(match.group(1).replace(",", ""))
            key = {"likes": "likes", "comments": "comment_count", "shares": "shares"}[
                match.group(2).lower()
            ]
            stats[key] = value
    return {
        "video_id": _video_id(video_url),
        "video_caption": caption,
        "hashtag": ",".join(_hashtags(caption)),
        "video_likes": stats["likes"],
        "video_comment_count": stats["comment_count"],
        "video_shares": stats["shares"],
    }


def _tiktok_card_payloads(driver: Any) -> list[dict[str, str]]:
    """Read visible comment text atomically; comment images are never inspected."""
    payloads = driver.execute_script(
        """
        const cards = [...document.querySelectorAll(arguments[0])];
        return cards.map((card) => {
          const oldWrapper = card.querySelector(
            'div[class*="DivCommentContentWrapper"]'
          );
          const contentElement =
            oldWrapper?.querySelector(':scope > span') ||
            card.querySelector('[data-e2e="comment-text"]') ||
            card.querySelector('p[data-e2e="comment-level-1"]') ||
            card.querySelector('p');
          const dateElement =
            card.querySelector(
              'div[class*="DivCommentSubContentWrapper"] span'
            ) ||
            card.querySelector('[data-e2e="comment-time"]') ||
            card.querySelector('time') ||
            card.querySelector('span[class*="SpanCreatedTime"]');
          const likeButton = card.querySelector('button[aria-label*="likes"]');
          return {
            content: (contentElement?.innerText || '').trim(),
            date_text: (dateElement?.innerText || '').trim(),
            like_label: likeButton?.getAttribute('aria-label') || '',
          };
        });
        """,
        TIKTOK_COMMENT_CARD_SELECTOR,
    )
    return [payload for payload in (payloads or []) if isinstance(payload, dict)]


def _tiktok_card_signature(
    payloads: Iterable[dict[str, str]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            str(payload.get("content") or "").strip(),
            str(payload.get("date_text") or "").strip(),
        )
        for payload in payloads
    )


def _parse_tiktok_cards(
    driver: Any,
    metadata: dict[str, object],
    discovery_tag: str,
    video_url: str,
    card_payloads: Iterable[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    rows = []
    payloads = (
        list(card_payloads)
        if card_payloads is not None
        else _tiktok_card_payloads(driver)
    )
    for payload in payloads:
        content = _text_only_comment(str(payload.get("content") or ""))
        date_text = str(payload.get("date_text") or "").strip()
        if not content:
            continue
        like_count = _likes(str(payload.get("like_label") or ""))
        rows.append(
            {
                "record_id": _stable_id(
                    "tiktok", str(metadata["video_id"]), date_text, content
                ),
                "source": "tiktok_comment",
                "content": content,
                "video_id": metadata["video_id"],
                "published_at": "",
                "comment_date_text": date_text,
                "likes": like_count,
                "language": "",
                "hashtag": metadata["hashtag"],
                "discovery_term": discovery_tag,
                "video_caption": metadata["video_caption"],
                "video_likes": metadata["video_likes"],
                "video_comment_count": metadata["video_comment_count"],
                "video_shares": metadata["video_shares"],
                "source_url": video_url,
                "retrieved_at": retrieved_at,
            }
        )
    return rows


def scrape_tiktok_video_comments(
    driver: Any,
    video_url: str,
    discovery_tag: str,
    max_comments: int = 100,
    verification_wait_seconds: int = 0,
) -> list[dict[str, object]]:
    """Collect public top-level comments, scrolling until stable or at the limit."""
    _, TimeoutException, _, By, EC, WebDriverWait = _selenium()
    from selenium.common.exceptions import (
        ElementClickInterceptedException,
        StaleElementReferenceException,
    )

    _safe_get(driver, video_url, verification_wait_seconds)

    def comment_entrypoint(current: Any) -> tuple[str, list[Any]] | bool:
        state = current.execute_script(
            """
            if (document.querySelector(arguments[0])) return 'cards';
            const emptyMarkers = arguments[1];
            const emptyState = [...document.querySelectorAll('div, p, span')]
              .some((element) => {
                const style = window.getComputedStyle(element);
                if (
                  style.display === 'none' ||
                  style.visibility === 'hidden' ||
                  element.getClientRects().length === 0
                ) return false;
                const text = (element.innerText || '').trim().toLowerCase();
                return emptyMarkers.some((marker) => text.includes(marker));
              });
            if (emptyState) return 'empty';
            if (document.querySelector(arguments[2])) return 'button';
            return '';
            """,
            TIKTOK_COMMENT_CARD_SELECTOR,
            list(TIKTOK_EMPTY_COMMENT_MARKERS),
            TIKTOK_COMMENT_BUTTON_SELECTOR,
        )
        return (state, []) if state else False

    def comment_content(current: Any) -> tuple[str, list[Any]] | bool:
        state = comment_entrypoint(current)
        return state if state and state[0] in {"cards", "empty"} else False

    entrypoint = _wait_for_tiktok_ui(
        driver,
        comment_entrypoint,
        "comments control",
        verification_wait_seconds,
    )
    entrypoint_kind, _ = entrypoint
    if entrypoint_kind == "empty":
        _log(f"TikTok: no public text comments available for {video_url}")
        return []
    if entrypoint_kind == "button":

        def clickable_comment_button(current: Any) -> Any:
            for button in current.find_elements(
                By.CSS_SELECTOR, TIKTOK_COMMENT_BUTTON_SELECTOR
            ):
                try:
                    if button.is_displayed() and button.is_enabled():
                        return button
                except StaleElementReferenceException:
                    continue
            return False

        last_click_error: Exception | None = None
        for attempt in range(3):
            comment_button = _wait_for_tiktok_ui(
                driver,
                clickable_comment_button,
                "clickable comments control",
                verification_wait_seconds,
            )
            try:
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", comment_button
                )
                comment_button.click()
                break
            except StaleElementReferenceException as error:
                last_click_error = error
                continue
            except ElementClickInterceptedException as error:
                last_click_error = error
                if attempt < 2:
                    continue
                driver.execute_script("arguments[0].click();", comment_button)
                break
        else:
            raise TikTokPageError(
                "TikTok comments control kept changing while it was being clicked."
            ) from last_click_error
        try:
            comments_tab = WebDriverWait(driver, 4).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//button[normalize-space()="Comments"]')
                )
            )
            comments_tab.click()
        except (
            TimeoutException,
            StaleElementReferenceException,
            ElementClickInterceptedException,
        ):
            pass

        comment_result = _wait_for_tiktok_ui(
            driver,
            comment_content,
            "comment cards or empty state",
            verification_wait_seconds,
        )
        if comment_result[0] == "empty":
            _log(f"TikTok: no public text comments available for {video_url}")
            return []

    metadata = _tiktok_metadata(driver, video_url)
    rows: list[dict[str, object]] = []
    stable_rounds = 0
    scroll_rounds = 0
    max_scroll_rounds = max(10, min(max_comments, 100))
    while (
        len(rows) < max_comments
        and stable_rounds < 3
        and scroll_rounds < max_scroll_rounds
    ):
        card_payloads = _tiktok_card_payloads(driver)
        previous_signature = _tiktok_card_signature(card_payloads)
        rows = _dedupe_rows(
            [
                *rows,
                *_parse_tiktok_cards(
                    driver,
                    metadata,
                    discovery_tag,
                    video_url,
                    card_payloads,
                ),
            ]
        )
        if len(rows) >= max_comments:
            break
        driver.execute_script(
            r"""
            const container = document.querySelector(
              'div[class*="DivCommentListContainer"], [data-e2e="comment-list"]'
            );
            if (container) {
              container.scrollTop = container.scrollHeight;
              container.dispatchEvent(new Event('scroll', {bubbles: true}));
            } else {
              window.scrollBy(0, 900);
            }
            """
        )
        scroll_rounds += 1
        try:
            WebDriverWait(driver, 4).until(
                lambda current: (
                    _tiktok_card_signature(_tiktok_card_payloads(current))
                    != previous_signature
                )
            )
        except TimeoutException:
            stable_rounds += 1
        else:
            stable_rounds = 0
    _log(
        f"TikTok: collected {len(rows[:max_comments])} text comments from "
        f"{video_url} in {scroll_rounds} scroll rounds"
    )
    return rows[:max_comments]


def discover_trip_product_urls(
    driver: Any,
    category_url: str,
    discovery_keyword: str,
    max_products: int = 60,
) -> dict[str, str]:
    """Discover relevant product URLs and titles from one Trip.com category."""
    if max_products <= 0:
        return {}
    _, TimeoutException, _, _, _, WebDriverWait = _selenium()
    _safe_get(driver, category_url)
    WebDriverWait(driver, 20).until(_trip_product_link_payloads)
    products: dict[str, str] = {}
    stable_rounds = 0
    discovery_rounds = 0
    max_discovery_rounds = 12
    while (
        len(products) < max_products
        and stable_rounds < 3
        and discovery_rounds < max_discovery_rounds
    ):
        payloads = _trip_product_link_payloads(driver)
        previous_signature = _trip_product_signature(payloads)
        for payload in payloads:
            href = str(payload.get("href") or "")
            title = str(payload.get("title") or "").strip()
            relevance = f"{discovery_keyword} {title}".lower()
            if _product_id(href) and any(
                term in relevance for term in TRIP_RELEVANCE_TERMS
            ):
                products[_canonical_url(href)] = title
                if len(products) >= max_products:
                    break
        if len(products) >= max_products:
            break
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        discovery_rounds += 1
        if discovery_rounds == 1 or discovery_rounds % 3 == 0:
            _log(
                f"Trip.com: {discovery_keyword}: {len(products)}/{max_products} "
                f"unique products after {discovery_rounds} scroll rounds"
            )
        try:
            WebDriverWait(driver, 4).until(
                lambda current: (
                    _trip_product_signature(_trip_product_link_payloads(current))
                    != previous_signature
                )
            )
        except TimeoutException:
            stable_rounds += 1
        else:
            stable_rounds = 0
    _log(
        f"Trip.com: discovered {len(products)} unique products in "
        f"{discovery_rounds} scroll rounds for {discovery_keyword}"
    )
    return dict(list(products.items())[:max_products])


def _trip_product_link_payloads(driver: Any) -> list[dict[str, str]]:
    payloads = driver.execute_script(
        """
        const products = new Map();
        for (const link of document.querySelectorAll(arguments[0])) {
          const href = link.href || link.getAttribute('href') || '';
          const title = (link.innerText || link.textContent || '').trim();
          if (!href) continue;
          if (!products.has(href) || (!products.get(href) && title)) {
            products.set(href, title);
          }
        }
        return [...products].map(([href, title]) => ({href, title}));
        """,
        TRIP_PRODUCT_LINK_SELECTOR,
    )
    return [payload for payload in (payloads or []) if isinstance(payload, dict)]


def _trip_product_signature(
    payloads: Iterable[dict[str, str]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                (
                    _canonical_url(str(payload.get("href") or "")),
                    str(payload.get("title") or "").strip(),
                )
                for payload in payloads
                if _product_id(str(payload.get("href") or ""))
            }
        )
    )


def _trip_review_card_payloads(driver: Any) -> list[dict[str, str]]:
    payloads = driver.execute_script(
        """
        return [...document.querySelectorAll(arguments[0])].map((card) => {
          const contentNode = card.querySelector(arguments[1]);
          const dateNode = card.querySelector(arguments[2]);
          return {
            content: (contentNode?.innerText || '').trim(),
            date_text: (
              dateNode?.getAttribute('datetime') || dateNode?.innerText || ''
            ).trim(),
          };
        });
        """,
        TRIP_REVIEW_CARD_SELECTOR,
        TRIP_REVIEW_CONTENT_SELECTOR,
        TRIP_REVIEW_DATE_SELECTOR,
    )
    return [payload for payload in (payloads or []) if isinstance(payload, dict)]


def _trip_review_payloads_ready(driver: Any) -> list[dict[str, str]]:
    payloads = [
        payload
        for payload in _trip_review_card_payloads(driver)
        if str(payload.get("content") or "").strip()
    ]
    if payloads:
        return payloads
    driver.execute_script(
        "window.scrollBy(0, Math.max(400, window.innerHeight * 0.7));"
    )
    return []


def _trip_review_signature(
    payloads: Iterable[dict[str, str]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            str(payload.get("content") or "").strip(),
            str(payload.get("date_text") or "").strip(),
        )
        for payload in payloads
    )


def _parse_trip_cards(
    driver: Any,
    product_url: str,
    product_title: str,
    discovery_keyword: str,
    rating: str,
    card_payloads: Iterable[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    product_id = _product_id(product_url)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    rows = []
    payloads = (
        list(card_payloads)
        if card_payloads is not None
        else _trip_review_card_payloads(driver)
    )
    for payload in payloads:
        content = str(payload.get("content") or "").strip()
        date_text = str(payload.get("date_text") or "").strip()
        if not content:
            continue
        rows.append(
            {
                "record_id": _stable_id("trip", product_id, date_text, content),
                "source": "trip_product_review",
                "content": content,
                "product_id": product_id,
                "product_title": product_title,
                "published_at": date_text,
                "rating": rating,
                "language": "en",
                "discovery_term": discovery_keyword,
                "source_url": product_url,
                "retrieved_at": retrieved_at,
            }
        )
    return rows


def _trip_product_title(driver: Any) -> str:
    return str(
        driver.execute_script(
            r"""
            const heading = document.querySelector('h1')?.innerText?.trim();
            if (heading) return heading;
            const metadata = document.querySelector('meta[property="og:title"]')
              ?.getAttribute('content')?.trim();
            return metadata || document.title.replace(/\s*\|\s*Trip\.com\s*$/, '');
            """
        )
        or ""
    ).strip()


def _open_trip_review_panel(driver: Any) -> str:
    return str(
        driver.execute_script(
            r"""
            if (document.querySelector(arguments[0])) return 'cards';
            const controls = [
              ...document.querySelectorAll('[class*="default_reviews__"]'),
              ...document.querySelectorAll('[class*="comment_score"]'),
            ];
            const reviewText = [...document.querySelectorAll('body *')].find((element) => {
              const text = (element.innerText || '').trim();
              return /^\(\d+\s+reviews?\)$/i.test(text);
            });
            const control = controls[0] || reviewText;
            if (!control) return 'missing';
            const clickable =
              control.closest('button, a, [role="button"]') ||
              control.parentElement ||
              control;
            clickable.scrollIntoView({block: 'center'});
            clickable.click();
            return 'clicked';
            """,
            TRIP_REVIEW_CARD_SELECTOR,
        )
        or ""
    )


def _open_trip_full_review_list(driver: Any) -> str:
    _, TimeoutException, _, By, _, WebDriverWait = _selenium()
    try:
        control = WebDriverWait(driver, 8).until(
            lambda current: next(
                (
                    element
                    for element in current.find_elements(
                        By.CSS_SELECTOR, TRIP_FULL_REVIEW_CONTROL_SELECTOR
                    )
                    if element.is_displayed()
                ),
                False,
            )
        )
    except TimeoutException:
        return "missing"
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", control)
    control.click()
    try:
        WebDriverWait(driver, 10).until(
            lambda current: current.find_elements(
                By.CSS_SELECTOR, TRIP_FULL_REVIEW_CARD_SELECTOR
            )
        )
    except TimeoutException:
        return "clicked"
    return "opened"


def _advance_trip_reviews(driver: Any) -> str:
    _, _, _, By, _, _ = _selenium()
    for control in driver.find_elements(
        By.CSS_SELECTOR, TRIP_NEXT_REVIEW_PAGE_SELECTOR
    ):
        if not control.is_displayed():
            continue
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center'});", control
        )
        control.click()
        return "clicked"
    return str(
        driver.execute_script(
            """
            const section = document.querySelector(arguments[0]);
            if (!section) return 'missing';
            const phrases = ['more', 'next', 'view all', 'show all'];
            const control = [...section.querySelectorAll('button, [role="button"]')]
              .find((element) => {
                const text = (element.innerText || '').trim().toLowerCase();
                const style = window.getComputedStyle(element);
                const visible =
                  style.display !== 'none' &&
                  style.visibility !== 'hidden' &&
                  element.getClientRects().length > 0;
                return visible && phrases.some((phrase) => text.includes(phrase));
              });
            if (control) {
              control.click();
              return 'clicked';
            }
            section.scrollIntoView({block: 'end'});
            return 'scrolled';
            """,
            TRIP_REVIEW_SECTION_SELECTOR,
        )
        or ""
    )


def _trip_review_class_summary(driver: Any) -> str:
    classes = driver.execute_script(
        """
        return [...new Set(
          [...document.querySelectorAll(
            '[class*="review"], [class*="Review"], '
            + '[class*="comment"], [class*="Comment"]'
          )].flatMap((element) => [...element.classList])
        )].filter((value) => /review|comment/i.test(value)).slice(0, 20);
        """
    )
    return ", ".join(str(value) for value in (classes or []))


def scrape_trip_product_reviews(
    driver: Any,
    product_url: str,
    discovery_keyword: str,
    max_reviews: int = 100,
) -> list[dict[str, object]]:
    """Collect public product reviews, expanding the review section when possible."""
    _, TimeoutException, _, _, _, WebDriverWait = _selenium()
    _safe_get(driver, product_url)
    wait = WebDriverWait(driver, 15)
    title = wait.until(_trip_product_title)
    relevance_text = f"{discovery_keyword} {title}".lower()
    if not any(term in relevance_text for term in TRIP_RELEVANCE_TERMS):
        raise ValueError(f"Trip.com product is outside the inclusion scope: {title}")
    panel_state = _open_trip_review_panel(driver)
    if panel_state == "missing":
        return []
    try:
        wait = WebDriverWait(driver, 30, poll_frequency=1)
        wait.until(_trip_review_payloads_ready)
    except TimeoutException as error:
        raise TripPageError(
            "Trip.com review text did not render after opening the review control "
            f'(title="{title[:120]}", classes="{_trip_review_class_summary(driver)}").'
        ) from error

    rating = driver.execute_script(
        """
        return (
          document.querySelector(
            '[class*="detail_review-score-score"], '
            + '[class*="review_section_score_value"], [class*="comment_score"]'
          )
            ?.innerText || ''
        ).trim();
        """
    )
    _open_trip_full_review_list(driver)
    rows: list[dict[str, object]] = []
    stable_rounds = 0
    review_rounds = 0
    max_review_rounds = max(5, min(max_reviews, 20))
    while (
        len(rows) < max_reviews
        and stable_rounds < 3
        and review_rounds < max_review_rounds
    ):
        card_payloads = _trip_review_card_payloads(driver)
        previous_signature = _trip_review_signature(card_payloads)
        rows = _dedupe_rows(
            [
                *rows,
                *_parse_trip_cards(
                    driver,
                    product_url,
                    title,
                    discovery_keyword,
                    rating,
                    card_payloads,
                ),
            ]
        )
        if len(rows) >= max_reviews:
            break
        action = _advance_trip_reviews(driver)
        if action == "missing":
            break
        review_rounds += 1
        if review_rounds == 1 or review_rounds % 5 == 0:
            _log(
                f"Trip.com: {len(rows)}/{max_reviews} reviews after "
                f"{review_rounds} expansion rounds for {product_url}"
            )
        try:
            WebDriverWait(driver, 4).until(
                lambda current: (
                    _trip_review_signature(_trip_review_card_payloads(current))
                    != previous_signature
                )
            )
        except TimeoutException:
            stable_rounds += 1
        else:
            stable_rounds = 0
    _log(
        f"Trip.com: collected {len(rows[:max_reviews])} reviews from {product_url} "
        f"in {review_rounds} expansion rounds"
    )
    return rows[:max_reviews]


def _parse_targets(values: Iterable[str], defaults: dict[str, str]) -> dict[str, str]:
    values = list(values)
    if not values:
        return defaults
    targets = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Target must use TERM=URL: {value}")
        term, url = value.split("=", 1)
        targets[term.strip()] = url.strip()
    return targets


def _error_row(source: str, term: str, url: str, error: Exception) -> dict[str, object]:
    return {
        "source": source,
        "term": term,
        "url": url,
        "error_type": type(error).__name__,
        "error": str(error).splitlines()[0][:500],
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }


def collect_tiktok(
    driver: Any,
    output_dir: Path,
    target_count: int,
    max_videos_per_tag: int,
    max_comments_per_video: int,
    explicit_targets: dict[str, str],
    verification_wait_seconds: int = 0,
) -> list[dict[str, object]]:
    output = output_dir / "tiktok_comments_aggregate.jsonl"
    error_output = output_dir.parent / "collection_errors.jsonl"
    rows = load_existing_rows(output_dir, "tiktok_comment")
    _log(f"TikTok: loaded {len(rows)} existing comments; target {target_count}")
    visited = {str(row.get("source_url") or "") for row in rows}
    candidates: list[tuple[str, str]] = list(explicit_targets.items())
    for tag in DEFAULT_TIKTOK_TAGS:
        if len(rows) >= target_count:
            break
        try:
            _log(f"TikTok: discovering up to {max_videos_per_tag} videos for {tag}")
            urls = discover_tiktok_video_urls(
                driver,
                tag,
                max_videos_per_tag,
                verification_wait_seconds,
            )
            candidates.extend((tag, url) for url in urls)
            _log(f"TikTok: discovered {len(urls)} videos for {tag}")
        except VerificationBarrierError:
            raise
        except Exception as error:
            write_error(_error_row("tiktok", tag, "", error), error_output)
            _log(f"TikTok: discovery failed for {tag} ({_brief_error(error)})")

    for tag, url in candidates:
        if len(rows) >= target_count:
            break
        canonical = _canonical_url(url)
        if canonical in visited:
            continue
        visited.add(canonical)
        try:
            _log(f"TikTok: scraping {canonical}")
            new_rows = scrape_tiktok_video_comments(
                driver,
                canonical,
                tag,
                max_comments_per_video,
                verification_wait_seconds,
            )
            rows = _dedupe_rows([*rows, *new_rows])
            write_jsonl(rows, output)
            _log(f"TikTok: {len(rows)}/{target_count} after {canonical}")
        except VerificationBarrierError:
            raise
        except Exception as error:
            write_error(_error_row("tiktok", tag, canonical, error), error_output)
            _log(f"TikTok: scrape failed for {canonical} ({_brief_error(error)})")
    write_jsonl(rows, output)
    return rows


def collect_trip(
    driver: Any,
    output_dir: Path,
    target_count: int,
    max_products_per_category: int,
    max_reviews_per_product: int,
    explicit_targets: dict[str, str],
) -> list[dict[str, object]]:
    output = output_dir / "trip_product_reviews_aggregate.jsonl"
    error_output = output_dir.parent / "collection_errors.jsonl"
    rows = load_existing_rows(output_dir, "trip_product_review")
    _log(f"Trip.com: loaded {len(rows)} existing reviews; target {target_count}")
    visited = {str(row.get("source_url") or "") for row in rows}
    candidates: list[tuple[str, str]] = list(explicit_targets.items())
    for keyword, category_url in DEFAULT_TRIP_CATEGORIES.items():
        if len(rows) >= target_count:
            break
        try:
            _log(
                "Trip.com: discovering up to "
                f"{max_products_per_category} products for {keyword}"
            )
            products = discover_trip_product_urls(
                driver, category_url, keyword, max_products_per_category
            )
            candidates.extend((keyword, url) for url in products)
            _log(f"Trip.com: discovered {len(products)} products for {keyword}")
        except Exception as error:
            write_error(
                _error_row("trip.com", keyword, category_url, error), error_output
            )
            _log(f"Trip.com: discovery failed for {keyword} ({_brief_error(error)})")

    for keyword, url in candidates:
        if len(rows) >= target_count:
            break
        canonical = _canonical_url(url)
        if canonical in visited:
            continue
        visited.add(canonical)
        try:
            _log(f"Trip.com: scraping {canonical}")
            new_rows = scrape_trip_product_reviews(
                driver, canonical, keyword, max_reviews_per_product
            )
            rows = _dedupe_rows([*rows, *new_rows])
            write_jsonl(rows, output)
            _log(f"Trip.com: {len(rows)}/{target_count} after {canonical}")
        except Exception as error:
            write_error(_error_row("trip.com", keyword, canonical, error), error_output)
            _log(f"Trip.com: scrape failed for {canonical} ({_brief_error(error)})")
    write_jsonl(rows, output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("all", "tiktok", "trip"),
        default="all",
        help="Collect both sources or only the selected platform",
    )
    parser.add_argument("--tiktok-target", action="append", default=[])
    parser.add_argument("--trip-target", action="append", default=[])
    parser.add_argument("--target-per-source", type=int, default=200)
    parser.add_argument("--max-videos-per-tag", type=int, default=12)
    parser.add_argument("--max-comments-per-video", type=int, default=50)
    parser.add_argument("--max-products-per-category", type=int, default=60)
    parser.add_argument("--max-reviews-per-product", type=int, default=50)
    parser.add_argument("--chrome-binary", type=Path, default=None)
    parser.add_argument("--chromedriver-path", type=Path, default=None)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Persistent Chrome profile (defaults to a source-specific data directory)",
    )
    parser.add_argument(
        "--verification-wait-seconds",
        type=int,
        default=180,
        help="How long headed Chrome waits for manual verification",
    )
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    if args.target_per_source < 1:
        parser.error("--target-per-source must be positive")
    if args.verification_wait_seconds < 0:
        parser.error("--verification-wait-seconds cannot be negative")
    if args.chrome_binary and not args.chrome_binary.is_file():
        parser.error(f"Chrome binary does not exist: {args.chrome_binary}")
    if args.chromedriver_path and not args.chromedriver_path.is_file():
        parser.error(f"ChromeDriver does not exist: {args.chromedriver_path}")
    collect_tiktok_source = args.source in {"all", "tiktok"}
    collect_trip_source = args.source in {"all", "trip"}
    profile_dir = args.profile_dir or Path(
        "data/selenium_trip_profile"
        if args.source == "trip"
        else "data/selenium_profile"
    )
    tiktok_targets = (
        _parse_targets(args.tiktok_target, DEFAULT_TIKTOK_TARGETS)
        if collect_tiktok_source
        else {}
    )
    trip_targets = (
        _parse_targets(args.trip_target, DEFAULT_TRIP_TARGETS)
        if collect_trip_source
        else {}
    )
    chromedriver_path = args.chromedriver_path or _default_chromedriver_path()
    verification_wait_seconds = args.verification_wait_seconds if args.headed else 0
    _log(
        "Starting browser: "
        f"Chrome={'custom binary' if args.chrome_binary else 'system default'}, "
        f"ChromeDriver={chromedriver_path or 'Selenium Manager'}, "
        f"profile={profile_dir.resolve()}"
    )
    if not args.headed and collect_tiktok_source:
        _log("Headless mode: manual verification is disabled; use --headed if prompted")
    driver = _driver(
        headless=not args.headed,
        chrome_binary=args.chrome_binary,
        chromedriver_path=chromedriver_path,
        profile_dir=profile_dir,
    )
    tiktok_rows: list[dict[str, object]] = []
    trip_rows: list[dict[str, object]] = []
    try:
        try:
            if collect_tiktok_source:
                tiktok_rows = collect_tiktok(
                    driver,
                    args.output_dir,
                    args.target_per_source,
                    args.max_videos_per_tag,
                    args.max_comments_per_video,
                    tiktok_targets,
                    verification_wait_seconds,
                )
            if collect_trip_source:
                trip_rows = collect_trip(
                    driver,
                    args.output_dir,
                    args.target_per_source,
                    args.max_products_per_category,
                    args.max_reviews_per_product,
                    trip_targets,
                )
        except VerificationBarrierError as error:
            _log(f"Collection stopped: {error}")
            raise SystemExit(2) from None
    finally:
        driver.quit()

    if args.source == "tiktok":
        _log(f"Complete: {len(tiktok_rows)} TikTok comments")
    elif args.source == "trip":
        _log(f"Complete: {len(trip_rows)} Trip.com reviews")
    else:
        _log(
            f"Complete: {len(tiktok_rows)} TikTok comments and "
            f"{len(trip_rows)} Trip.com reviews"
        )


if __name__ == "__main__":
    main()
