import base64

from lib.url_unwrap import is_known_tracker, unwrap_url


def test_noop_when_no_query():
    url = "https://example.com/job/123"
    assert unwrap_url(url) == url


def test_noop_when_query_has_no_target_param():
    url = "https://example.com/job/123?ref=abc&src=feed"
    assert unwrap_url(url) == url


def test_google_url_redirect():
    real = "https://careers.example.com/job/123"
    wrapped = f"https://www.google.com/url?q={real}&sa=U"
    assert unwrap_url(wrapped) == real


def test_indeed_rd_style_url_param():
    real = "https://jobs.example.com/posting/456"
    wrapped = f"https://click.indeed.com/rd?url={real}&tk=xyz"
    assert unwrap_url(wrapped) == real


def test_urlencoded_target():
    from urllib.parse import quote
    real = "https://example.com/job?x=1&y=2"
    wrapped = f"https://tracker.test/click?u={quote(real, safe='')}"
    assert unwrap_url(wrapped) == real


def test_base64_encoded_target():
    real = "https://example.com/job/789"
    encoded = base64.urlsafe_b64encode(real.encode()).decode().rstrip("=")
    wrapped = f"https://tracker.test/c?u={encoded}"
    assert unwrap_url(wrapped) == real


def test_recursive_unwrap():
    real = "https://careers.example.com/final"
    inner = f"https://click.indeed.com/rd?url={real}"
    from urllib.parse import quote
    outer = f"https://click.appcast.io/track?redirect={quote(inner, safe='')}"
    assert unwrap_url(outer) == real


def test_depth_limit_terminates():
    # Self-referential loop shouldn't hang
    url = "https://t.test/r?url=https://t.test/r?url=https://t.test/r"
    # Just assert it returns something without raising
    assert unwrap_url(url, max_depth=3)


def test_known_tracker_detection():
    assert is_known_tracker("https://click.indeed.com/rd?url=x")
    assert not is_known_tracker("https://example.com/job")


def test_empty_url():
    assert unwrap_url("") == ""
