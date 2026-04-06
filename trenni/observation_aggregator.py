"""Observation event aggregator for autonomous optimization loop.

Periodically queries pasloe for observation.* events, aggregates by metric_type,
and spawns optimizer tasks when thresholds are exceeded.

Per ADR-0010 extension for Factorio Tool Evolution MVP.

Key semantics:
- window_count: count of ALL observation events in the window (for threshold check)
- new_ids: IDs of events NOT YET processed (for dedup, spawn control, and hash)

This separation ensures:
- Threshold is based on total window activity (not just "new this round")
- Dedup prevents duplicate spawns for same batch of new events
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import httpx
import logging

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    metric_type: str
    count: int  # Window-wide count (ALL events in window, for threshold)
    threshold: float
    exceeded: bool
    role: str | None = None


async def aggregate_observations(
    pasloe_url: str,
    window_hours: int,
    thresholds: dict[str, float],
    processed_ids: set[str] | None = None,
) -> tuple[list[AggregationResult], list[str]]:
    """Query pasloe for observation.* events in window, aggregate by metric_type.
    
    Returns TWO things with distinct semantics:
    
    1. results[].count: Window-wide total (ALL observation events in window)
       - Used for threshold check ("5 occurrences in 24h window")
       - NOT affected by processed_ids filtering
       
    2. new_ids: IDs of events NOT yet in processed_ids
       - Used for dedup (hash-based synthetic event id)
       - Used for spawn decision (only spawn if there are new events)
       - Empty if all window events were already processed
    
    This separation is critical for correct threshold behavior:
    - If threshold=5, and events arrive: round1=3, round2=3
    - round1: count=3, new_ids=[e1,e2,e3], not exceeded
    - round2: count=6 (window total), new_ids=[e4,e5,e6], exceeded → spawn optimizer
    
    Args:
        pasloe_url: Pasloe API base URL
        window_hours: How many hours back to query
        thresholds: Threshold dict (metric_type -> threshold value)
        processed_ids: Set of already-processed observation event IDs
        
    Returns:
        (results, new_ids) tuple:
        - results: AggregationResult with window-wide counts
        - new_ids: List of event IDs not yet processed
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    all_events = []  # ALL observation events in window (for count)
    all_event_ids: list[str] = []  # ALL IDs in window
    cursor = None
    
    async with httpx.AsyncClient() as client:
        while True:
            params = {"since": cutoff.isoformat(), "limit": 1000, "order": "asc"}
            if cursor:
                params["cursor"] = cursor
            
            resp = await client.get(f"{pasloe_url}/events", params=params)
            resp.raise_for_status()
            batch = resp.json()
            
            # Collect ALL observation.* events in window (no filtering yet)
            for evt in batch:
                if evt.get("type", "").startswith("observation."):
                    all_events.append(evt)
                    event_id = evt.get("id", "")
                    if event_id:
                        all_event_ids.append(event_id)
            
            cursor = resp.headers.get("X-Next-Cursor")
            if not cursor:
                break
    
    # Compute window-wide counts (ALL events, NOT filtered by processed_ids)
    counts: dict[str, int] = {}
    for evt in all_events:
        event_type = evt.get("type", "")
        if not event_type.startswith("observation."):
            continue
        metric = event_type.split(".", 1)[1] if "." in event_type else ""
        counts[metric] = counts.get(metric, 0) + 1
    
    # Build results with window-wide counts
    results = []
    for metric, count in counts.items():
        threshold = thresholds.get(metric, float("inf"))
        results.append(AggregationResult(
            metric_type=metric,
            count=count,  # Window-wide total (for threshold)
            threshold=threshold,
            exceeded=(count >= threshold),
        ))
    
    # Compute new_ids (events NOT yet processed) - for dedup/spawn control
    processed_set = processed_ids or set()
    new_ids = [id for id in all_event_ids if id not in processed_set]
    
    return results, new_ids