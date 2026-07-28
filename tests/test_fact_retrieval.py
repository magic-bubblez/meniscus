from __future__ import annotations

import config
import fact_retrieval as fr
from fact_retrieval import Fact


def _add_event(conn, content, timestamp, source="test"):
    return int(
        conn.execute(
            "INSERT INTO events (source, content, timestamp, extraction_status, content_hash) "
            "VALUES (?, ?, ?, 'completed', ?)",
            (source, content, timestamp, f"{source}:{content}:{timestamp}"),
        ).lastrowid
    )


def _add_entity(conn, name):
    return int(
        conn.execute(
            "INSERT INTO entities (canonical_name, normalized_form, created_at) "
            "VALUES (?, ?, '2026-01-01')",
            (name, name),
        ).lastrowid
    )


def _add_fact(conn, event_id, text, position, entity_ids):
    extraction_id = int(
        conn.execute(
            "INSERT INTO extractions (event_id, provider, model, prompt_version, extracted_at) "
            "VALUES (?, 'p', 'm', 'v', '2026-01-01')",
            (event_id,),
        ).lastrowid
    )
    fact_id = int(
        conn.execute(
            "INSERT INTO facts (event_id, extraction_id, text, position, created_at) "
            "VALUES (?, ?, ?, ?, '2026-01-01')",
            (event_id, extraction_id, text, position),
        ).lastrowid
    )
    for entity_id in entity_ids:
        conn.execute(
            "INSERT INTO fact_entity_edges (fact_id, entity_id) VALUES (?, ?)",
            (fact_id, entity_id),
        )
    return fact_id


def _seed(conn):
    jwt, ttl, groceries = _add_entity(conn, "jwt"), _add_entity(conn, "ttl"), _add_entity(conn, "groceries")
    e1 = _add_event(conn, "jwt work", "2026-03-01T10:00:00+00:00")
    f1 = _add_fact(conn, e1, "The jwt refresh token was failing.", 0, [jwt])
    f2 = _add_fact(conn, e1, "The ttl mismatch caused the jwt failure.", 1, [jwt, ttl])
    e2 = _add_event(conn, "groceries", "2026-03-01T11:00:00+00:00")
    f3 = _add_fact(conn, e2, "The person bought groceries.", 0, [groceries])
    conn.commit()
    return {"f1": f1, "f2": f2, "f3": f3}


def test_text_door_matches_facts(conn):
    ids = _seed(conn)
    matched = {f.fact_id for f in fr.search_facts(conn, text="jwt")}
    assert matched == {ids["f1"], ids["f2"]}


def test_entity_door_matches_facts(conn):
    ids = _seed(conn)
    matched = fr.search_facts(conn, entity="ttl")
    assert [f.fact_id for f in matched] == [ids["f2"]]
    assert matched[0].matched_by == ["entity"]


def test_retrieve_gathers_shared_entity_neighbour_chronological(conn, monkeypatch):
    monkeypatch.setattr(config, "SOURCE_COLLAPSE", False)  # isolate neighbour-gather from collapse
    monkeypatch.setattr(config, "NEIGHBOUR_ENABLED", True)  # neighbours are opt-in now
    ids = _seed(conn)
    facts = fr.retrieve(conn, entity="ttl")
    by_id = {f.fact_id: f for f in facts}
    # f2 matched directly; f1 gathered as a neighbour (shares the jwt entity).
    assert by_id[ids["f2"]].matched is True
    assert by_id[ids["f1"]].matched is False
    # groceries fact shares nothing -> not gathered.
    assert ids["f3"] not in by_id
    # chronological (f1 at position 0 precedes f2 at position 1, same event time).
    assert [f.fact_id for f in facts] == sorted(by_id, key=lambda i: (by_id[i].timestamp, i))


def test_retrieve_time_window_filters(conn):
    _seed(conn)
    assert fr.search_facts(conn, text="jwt", start="2026-04-01") == []


def _scored(scores):
    return [
        Fact(fact_id=i, event_id=1, text="x", timestamp="2026-01-01", source="t", scores={"neighbour": s})
        for i, s in enumerate(scores)
    ]


def test_cut_ratio_and_gap_diverge(monkeypatch):
    scored = _scored([0.9, 0.7, 0.68, 0.66])

    monkeypatch.setattr(config, "NEIGHBOUR_CUT_RULE", "ratio")
    monkeypatch.setattr(config, "NEIGHBOUR_CUT_RATIO", 0.75)
    ratio_kept = fr._apply_cut(list(scored))
    assert [f.fact_id for f in ratio_kept] == [0, 1, 2]  # >= 0.675

    monkeypatch.setattr(config, "NEIGHBOUR_CUT_RULE", "gap")
    gap_kept = fr._apply_cut(list(scored))
    assert [f.fact_id for f in gap_kept] == [0]  # largest drop is 0.9 -> 0.7


def test_cut_respects_neighbour_max(monkeypatch):
    scored = _scored([0.9] * 30)
    monkeypatch.setattr(config, "NEIGHBOUR_CUT_RULE", "ratio")
    monkeypatch.setattr(config, "NEIGHBOUR_CUT_RATIO", 0.5)
    monkeypatch.setattr(config, "NEIGHBOUR_MAX", 20)
    assert len(fr._apply_cut(scored)) == 20


def test_source_collapse_replaces_event_when_facts_dominate(conn, monkeypatch):
    monkeypatch.setattr(config, "SOURCE_COLLAPSE", True)
    ids = _seed(conn)
    items = fr.retrieve(conn, text="jwt")
    # e1's two facts (18 tokens) cost more than the raw "jwt work" (0 tokens) -> collapse.
    assert [i.kind for i in items] == [fr.KIND_EVENT]
    assert items[0].text == "jwt work"
    assert items[0].event_id == conn.execute(
        "SELECT event_id FROM facts WHERE id = ?", (ids["f1"],)
    ).fetchone()[0]


def test_source_collapse_keeps_facts_on_narrow_query(conn, monkeypatch):
    monkeypatch.setattr(config, "SOURCE_COLLAPSE", True)
    kafka = _add_entity(conn, "kafka")
    long_content = "A very long note about kafka and many other unrelated matters " * 6
    e = _add_event(conn, long_content, "2026-05-01T10:00:00+00:00")
    kept = _add_fact(conn, e, "Kafka rebalancing stalled.", 0, [kafka])
    conn.commit()
    items = fr.retrieve(conn, text="kafka")
    # one short fact (~6 tokens) against a long raw event -> stays a fact, not collapsed.
    assert [i.kind for i in items] == [fr.KIND_FACT]
    assert items[0].fact_id == kept


def test_source_collapse_flag_off_returns_facts(conn, monkeypatch):
    monkeypatch.setattr(config, "SOURCE_COLLAPSE", False)
    ids = _seed(conn)
    items = fr.retrieve(conn, text="jwt")
    assert {i.kind for i in items} == {fr.KIND_FACT}
    assert {i.fact_id for i in items} == {ids["f1"], ids["f2"]}


def test_episode_groups_to_anchor_session(conn):
    a = _add_event(conn, "standup", "2026-03-01T10:00:00+00:00")
    b = _add_event(conn, "auth bug", "2026-03-01T10:00:05+00:00")
    _add_event(conn, "afternoon session", "2026-03-01T13:00:00+00:00")  # 3h gap, in window
    conn.commit()
    episodes = fr.episode(conn, anchor_event_id=a)
    assert len(episodes) == 1
    assert [ev.event_id for ev in episodes[0].events] == [a, b]


def test_episode_browse_returns_all_sessions_in_window(conn):
    a = _add_event(conn, "s1a", "2026-03-01T10:00:00+00:00")
    _add_event(conn, "s1b", "2026-03-01T10:00:05+00:00")
    _add_event(conn, "s2", "2026-03-01T13:00:00+00:00")
    conn.commit()
    episodes = fr.episode(conn, anchor_event_id=a, group=False)
    assert [e.event_count for e in episodes] == [2, 1]


def test_episode_no_anchor_returns_empty(conn):
    assert fr.episode(conn, anchor_text="nothing was ever recorded") == []
