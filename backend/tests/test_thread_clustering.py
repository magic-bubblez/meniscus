from datetime import datetime, timedelta, timezone

from app.pipeline import get_thread_subgraph, ingest_event

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _iso(minutes: int) -> str:
    return (BASE + timedelta(minutes=minutes)).isoformat()


def test_follow_up_reconnects_to_its_own_thread_not_the_last_touched_one():
    a1 = ingest_event(
        {
            "source": "github",
            "content": "Refactored the JWT refresh token middleware",
            "timestamp": _iso(0),
        }
    )
    b1 = ingest_event(
        {
            "source": "notion",
            "content": "Drafted the fundraising pitch deck outline",
            "timestamp": _iso(40),
        }
    )
    a2 = ingest_event(
        {
            "source": "chatgpt",
            "content": "Asked about JWT refresh token expiry edge cases",
            "timestamp": _iso(45),
        }
    )

    assert a1["thread_id"] != b1["thread_id"]
    assert a2["thread_id"] == a1["thread_id"]
    assert a2["thread_id"] != b1["thread_id"]

    thread = get_thread_subgraph(a1["thread_id"])
    assert len(thread["events"]) == 2


def test_unrelated_events_close_in_time_still_split():
    first = ingest_event(
        {
            "source": "youtube",
            "content": "Watched a video about sourdough starter hydration",
            "timestamp": _iso(0),
        }
    )
    second = ingest_event(
        {
            "source": "github",
            "content": "Committed a fix for the database migration script",
            "timestamp": _iso(2),
        }
    )

    assert first["thread_id"] != second["thread_id"]


def test_short_signal_less_follow_up_continues_the_recent_thread():
    first = ingest_event(
        {
            "source": "chatgpt",
            "content": "Asked about JWT refresh token expiry edge cases",
            "timestamp": _iso(0),
        }
    )
    reply = ingest_event(
        {
            "source": "chatgpt",
            "content": "ok",
            "timestamp": _iso(3),
        }
    )

    assert reply["thread_id"] == first["thread_id"]


def test_related_events_far_apart_in_time_still_cluster_on_topic():
    first = ingest_event(
        {
            "source": "youtube",
            "content": "JWT authentication overview: access token and refresh mechanism",
            "timestamp": _iso(0),
        }
    )
    second = ingest_event(
        {
            "source": "chatgpt",
            "content": "Why is my access token expiring immediately after login",
            "timestamp": _iso(6 * 60),
        }
    )

    assert second["thread_id"] == first["thread_id"]
