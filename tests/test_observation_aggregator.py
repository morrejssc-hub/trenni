"""Tests for observation aggregator (ADR-0010 extension)."""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from trenni.observation_aggregator import aggregate_observations, AggregationResult


def make_response(events, cursor=None):
    """Helper to create a mock httpx.Response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = events
    response.headers = {"X-Next-Cursor": cursor} if cursor else {}
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def sample_events():
    """Sample observation events for testing."""
    now = datetime.now(timezone.utc)
    return [
        {
            "id": "evt-1",
            "source_id": "palimpsest-agent",
            "type": "observation.tool_repetition",
            "ts": now.isoformat(),
            "data": {
                "job_id": "job-1",
                "task_id": "task-1",
                "role": "worker",
                "team": "factorio",
                "tool_name": "factorio_call_script(actions.place)",
                "call_count": 10,
                "arg_pattern": "grid_5x2",
                "similarity": 0.85,
            },
        },
        {
            "id": "evt-2",
            "source_id": "palimpsest-agent",
            "type": "observation.tool_repetition",
            "ts": (now - timedelta(minutes=5)).isoformat(),
            "data": {
                "job_id": "job-2",
                "task_id": "task-2",
                "role": "worker",
                "team": "factorio",
                "tool_name": "factorio_call_script(actions.place)",
                "call_count": 8,
                "arg_pattern": "grid_3x3",
                "similarity": 0.90,
            },
        },
        {
            "id": "evt-3",
            "source_id": "palimpsest-agent",
            "type": "observation.budget_variance",
            "ts": (now - timedelta(hours=1)).isoformat(),
            "data": {
                "task_id": "task-3",
                "job_id": "job-3",
                "role": "planner",
                "estimated_budget": 1.0,
                "actual_cost": 1.5,
                "variance_ratio": 0.5,
            },
        },
    ]


@pytest.mark.asyncio
async def test_aggregate_below_threshold(sample_events):
    """Test aggregation when count is below threshold."""
    tool_repetition_events = [e for e in sample_events if e["type"] == "observation.tool_repetition"][:1]
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response(tool_repetition_events))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results, new_ids = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 5.0},
        )
    
    assert len(results) == 1
    assert results[0].metric_type == "tool_repetition:factorio_call_script(actions.place)"
    assert results[0].count == 1
    assert not results[0].exceeded
    assert "tool_repetition:factorio_call_script(actions.place)" in new_ids
    assert new_ids["tool_repetition:factorio_call_script(actions.place)"] == ["evt-1"]


@pytest.mark.asyncio
async def test_aggregate_exceeds_threshold(sample_events):
    """Test aggregation when count exceeds threshold."""
    tool_repetition_events = [e for e in sample_events if e["type"] == "observation.tool_repetition"]
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response(tool_repetition_events))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results, new_ids = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 2.0},
        )
    
    assert len(results) == 1
    assert results[0].metric_type == "tool_repetition:factorio_call_script(actions.place)"
    assert results[0].count == 2
    assert results[0].exceeded
    assert "tool_repetition:factorio_call_script(actions.place)" in new_ids
    assert set(new_ids["tool_repetition:factorio_call_script(actions.place)"]) == {"evt-1", "evt-2"}


@pytest.mark.asyncio
async def test_aggregate_multiple_metric_types(sample_events):
    """Test aggregation with multiple metric types."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response(sample_events))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results, new_ids = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 1.0, "budget_variance": 0.3},
        )
    
    assert len(results) == 2
    metrics = {r.metric_type: r for r in results}
    
    assert "tool_repetition:factorio_call_script(actions.place)" in metrics
    assert metrics["tool_repetition:factorio_call_script(actions.place)"].count == 2
    assert metrics["tool_repetition:factorio_call_script(actions.place)"].exceeded
    
    assert "budget_variance" in metrics
    assert metrics["budget_variance"].count == 1
    assert metrics["budget_variance"].exceeded
    
    assert "tool_repetition:factorio_call_script(actions.place)" in new_ids
    assert "budget_variance" in new_ids
    assert set(new_ids["tool_repetition:factorio_call_script(actions.place)"]) == {"evt-1", "evt-2"}
    assert set(new_ids["budget_variance"]) == {"evt-3"}


@pytest.mark.asyncio
async def test_aggregate_empty_response():
    """Test aggregation with no events."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response([]))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results, new_ids = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 5.0},
        )
    
    assert len(results) == 0
    assert new_ids == {}  # Empty dict when no events


@pytest.mark.asyncio
async def test_aggregate_pagination():
    """Test aggregation with paginated responses."""
    batch1 = [{"id": "evt-1", "type": "observation.tool_repetition", "ts": "2024-01-01T00:00:00Z", "data": {}}]
    batch2 = [{"id": "evt-2", "type": "observation.tool_repetition", "ts": "2024-01-01T00:01:00Z", "data": {}}]
    
    call_count = 0
    async def mock_get(url, params, headers=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return make_response(batch1, cursor="cursor-abc")
        else:
            return make_response(batch2)
    
    mock_client = MagicMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results, new_ids = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 1.0},
        )
    
    assert len(results) == 1
    assert results[0].count == 2
    assert call_count == 2
    assert "tool_repetition:unknown" in new_ids
    assert set(new_ids["tool_repetition:unknown"]) == {"evt-1", "evt-2"}


@pytest.mark.asyncio
async def test_aggregate_filters_non_observation_events(sample_events):
    """Test that only observation.* events are counted."""
    mixed_events = sample_events + [
        {"id": "evt-4", "type": "tool.exec", "ts": "2024-01-01T00:00:00Z", "data": {}},
        {"id": "evt-5", "type": "agent.job.completed", "ts": "2024-01-01T00:01:00Z", "data": {}},
    ]
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response(mixed_events))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results, new_ids = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 1.0, "budget_variance": 0.3},
        )
    
    assert len(results) == 2
    metrics = {r.metric_type: r for r in results}
    assert "tool_repetition:factorio_call_script(actions.place)" in metrics
    assert "budget_variance" in metrics
    assert "tool.exec" not in metrics
    assert "tool_repetition:factorio_call_script(actions.place)" in new_ids
    assert "budget_variance" in new_ids
    assert set(new_ids["tool_repetition:factorio_call_script(actions.place)"]) == {"evt-1", "evt-2"}
    assert set(new_ids["budget_variance"]) == {"evt-3"}


def test_aggregation_result_dataclass():
    """Test AggregationResult dataclass."""
    result = AggregationResult(
        metric_type="tool_repetition",
        count=10,
        threshold=5.0,
        exceeded=True,
        evidence=[{"bundle": "factorio", "tool_name": "test_tool"}],
        role="worker",
    )
    
    assert result.metric_type == "tool_repetition"
    assert result.count == 10
    assert result.threshold == 5.0
    assert result.exceeded is True
    assert result.evidence == [{"bundle": "factorio", "tool_name": "test_tool"}]
    assert result.role == "worker"
    
    result2 = AggregationResult(
        metric_type="tool_repetition",
        count=3,
        threshold=5.0,
        exceeded=False,
        evidence=[],
    )
    assert not result2.exceeded
    assert result2.evidence == []


@pytest.mark.asyncio
async def test_aggregate_skips_processed_ids(sample_events):
    """processed_ids only affects new_ids, NOT window count."""
    tool_repetition_events = [e for e in sample_events if e["type"] == "observation.tool_repetition"]
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response(tool_repetition_events))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results1, new_ids1 = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 2.0},
        )
    
    assert results1[0].count == 2
    assert "tool_repetition:factorio_call_script(actions.place)" in new_ids1
    assert set(new_ids1["tool_repetition:factorio_call_script(actions.place)"]) == {"evt-1", "evt-2"}
    
    processed_ids = set(new_ids1["tool_repetition:factorio_call_script(actions.place)"])
    with patch("httpx.AsyncClient", return_value=mock_client):
        results2, new_ids2 = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 2.0},
            processed_ids=processed_ids,
        )
    
    assert len(results2) == 1
    assert results2[0].count == 2
    assert results2[0].exceeded
    assert "tool_repetition:factorio_call_script(actions.place)" not in new_ids2  # All IDs processed


@pytest.mark.asyncio
async def test_aggregate_partial_processed_ids(sample_events):
    """processed_ids only affects new_ids, window count includes ALL events."""
    all_events = [e for e in sample_events if e["type"] == "observation.tool_repetition"]
    all_events.append({
        "id": "evt-3",
        "type": "observation.tool_repetition",
        "ts": "2024-01-01T00:02:00Z",
        "data": {},
    })
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response(all_events))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    processed_ids = {"evt-1"}
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results, new_ids = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 2.0},
            processed_ids=processed_ids,
        )
    
    assert len(results) == 2
    # factorio_call_script has count=2 (from sample_events evt-1, evt-2)
    factorio_result = [r for r in results if r.metric_type == "tool_repetition:factorio_call_script(actions.place)"][0]
    assert factorio_result.count == 2
    assert factorio_result.exceeded
    # unknown has count=1 (from evt-3 added without tool_name)
    unknown_result = [r for r in results if r.metric_type == "tool_repetition:unknown"][0]
    assert unknown_result.count == 1
    assert not unknown_result.exceeded
    assert "tool_repetition:factorio_call_script(actions.place)" in new_ids
    assert set(new_ids["tool_repetition:factorio_call_script(actions.place)"]) == {"evt-2"}
    assert "tool_repetition:unknown" in new_ids
    assert new_ids["tool_repetition:unknown"] == ["evt-3"]


@pytest.mark.asyncio
async def test_api_key_header_is_x_api_key():
    """API key should be sent via X-API-Key header, not Authorization: Bearer."""
    events = [{"id": "evt-1", "type": "observation.tool_repetition", "ts": "2024-01-01T00:00:00Z", "data": {}}]
    
    captured_headers = None
    async def mock_get(url, params, headers=None):
        nonlocal captured_headers
        captured_headers = headers
        return make_response(events)
    
    mock_client = MagicMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 1.0},
            api_key="test-api-key",
        )
    
    assert captured_headers is not None
    assert captured_headers.get("X-API-Key") == "test-api-key"
    assert "Authorization" not in captured_headers


@pytest.mark.asyncio
async def test_no_api_key_means_no_header():
    """When api_key is empty, no auth header should be sent."""
    events = [{"id": "evt-1", "type": "observation.tool_repetition", "ts": "2024-01-01T00:00:00Z", "data": {}}]
    
    captured_headers = None
    async def mock_get(url, params, headers=None):
        nonlocal captured_headers
        captured_headers = headers
        return make_response(events)
    
    mock_client = MagicMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 1.0},
            api_key="",  # No API key
        )
    
    assert captured_headers is not None
    assert "X-API-Key" not in captured_headers
    assert "Authorization" not in captured_headers


@pytest.mark.asyncio
async def test_window_count_semantics_for_threshold():
    """Threshold uses window-wide total, not just 'new this round'."""
    round1_events = [
        {"id": "evt-1", "type": "observation.tool_repetition", "ts": "2024-01-01T00:00:00Z", "data": {}},
        {"id": "evt-2", "type": "observation.tool_repetition", "ts": "2024-01-01T00:01:00Z", "data": {}},
        {"id": "evt-3", "type": "observation.tool_repetition", "ts": "2024-01-01T00:02:00Z", "data": {}},
    ]
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response(round1_events))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results1, new_ids1 = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 5.0},
        )
    
    assert results1[0].count == 3
    assert not results1[0].exceeded
    assert "tool_repetition:unknown" in new_ids1
    assert set(new_ids1["tool_repetition:unknown"]) == {"evt-1", "evt-2", "evt-3"}
    
    round2_events = round1_events + [
        {"id": "evt-4", "type": "observation.tool_repetition", "ts": "2024-01-01T00:03:00Z", "data": {}},
        {"id": "evt-5", "type": "observation.tool_repetition", "ts": "2024-01-01T00:04:00Z", "data": {}},
        {"id": "evt-6", "type": "observation.tool_repetition", "ts": "2024-01-01T00:05:00Z", "data": {}},
    ]
    
    mock_client.get = AsyncMock(return_value=make_response(round2_events))
    processed_ids = set(new_ids1["tool_repetition:unknown"])
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results2, new_ids2 = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 5.0},
            processed_ids=processed_ids,
        )
    
    assert results2[0].count == 6
    assert results2[0].exceeded
    assert "tool_repetition:unknown" in new_ids2
    assert set(new_ids2["tool_repetition:unknown"]) == {"evt-4", "evt-5", "evt-6"}

@pytest.mark.asyncio
async def test_evidence_extraction_from_events():
    """AggregationResult includes evidence payloads from observation events.
    
    Per plan Task 1.1: extract latest 5 events with tool_name, arg_pattern, bundle.
    """
    now = datetime.now(timezone.utc)
    events = [
        {
            "id": f"evt-{i}",
            "type": "observation.tool_repetition",
            "ts": (now - timedelta(minutes=i)).isoformat(),
            "data": {
                "role": "worker",
                "bundle": "factorio",
                "tool_name": "factorio_call_script(find_ore_basic)",
                "call_count": 5 + i,
                "arg_pattern": "find_ore_basic",
                "similarity": 0.8,
            },
        }
        for i in range(1, 8)  # 7 events
    ]
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response(events))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results, new_ids = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 1.0},
        )
    
    assert len(results) == 1
    assert results[0].metric_type == "tool_repetition:factorio_call_script(find_ore_basic)"
    
    # Evidence should be latest 5 (sorted by ts, newest first)
    evidence = results[0].evidence
    assert len(evidence) == 5
    
    # Latest event (evt-1) should be first
    assert evidence[0]["bundle"] == "factorio"
    assert evidence[0]["tool_name"] == "factorio_call_script(find_ore_basic)"
    assert evidence[0]["arg_pattern"] == "find_ore_basic"
    assert evidence[0]["call_count"] == 6  # 5 + 1
    assert evidence[0]["similarity"] == 0.8
    
    # All evidence should have bundle=factorio
    for e in evidence:
        assert e["bundle"] == "factorio"
        assert "tool_name" in e
        assert "arg_pattern" in e


@pytest.mark.asyncio
async def test_evidence_empty_when_no_events():
    """Evidence is empty list when no observation events.
    
    Per plan Task 1.5: evidence empty leads to bundle fallback.
    """
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response([]))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results, new_ids = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 1.0},
        )
    
    assert len(results) == 0
    assert new_ids == {}  # Empty dict when no events


@pytest.mark.asyncio
async def test_evidence_missing_bundle_field():
    """Events without bundle field are included but have empty bundle.
    
    Per plan: _resolve_bundle_for_observations handles missing bundle.
    """
    now = datetime.now(timezone.utc)
    events = [
        {
            "id": "evt-1",
            "type": "observation.tool_repetition",
            "ts": now.isoformat(),
            "data": {
                "role": "worker",
                # bundle field missing
                "tool_name": "test_tool",
                "call_count": 10,
            },
        },
    ]
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=make_response(events))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        results, new_ids = await aggregate_observations(
            "http://localhost:8000",
            window_hours=24,
            thresholds={"tool_repetition": 1.0},
        )
    
    assert len(results) == 1
    evidence = results[0].evidence
    assert len(evidence) == 1
    assert evidence[0].get("bundle") == ""  # Missing bundle = empty string
