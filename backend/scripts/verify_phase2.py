"""
Verify Phase 2: invoke start-analytics for a phase1_complete video and assert
the merged outputs preserve Phase 1 data exactly.

Usage:
    python -m backend.scripts.verify_phase2 --video-id <id>

Preconditions:
- The video row is in status 'phase1_complete' (or 'failed_phase2' for retry).
- `results_storage_path` is populated.
- Env vars: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.

This script snapshots the Phase 1 results, kicks off Phase 2 through the
`start-analytics` Edge Function, polls until completion, and then asserts that
Phase 2 added analytics/skeleton data without mutating any Phase 1 keys.
"""
from __future__ import annotations

import argparse
import json
import os
import time


# Import lazily so the script can be imported without Supabase env vars.
def _supabase_client():
    import sys
    from pathlib import Path
    backend_dir = str(Path(__file__).resolve().parents[1])
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from supabase_helpers import supabase_client  # type: ignore
    return supabase_client()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, help="UUID of the video to verify")
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=1800,
        help="Seconds to wait for Phase 2 to terminate (default: 1800)",
    )
    args = parser.parse_args()

    import requests  # local import to keep top-level import surface small

    sb = _supabase_client()

    # Sanity: precondition.
    row = (
        sb.table("videos")
        .select("status, results_storage_path")
        .eq("id", args.video_id)
        .single()
        .execute()
        .data
    )
    assert row, f"Video {args.video_id} not found"
    assert row["status"] in ("phase1_complete", "failed_phase2"), (
        f"Bad starting status: {row['status']} "
        f"(expected 'phase1_complete' or 'failed_phase2')"
    )
    assert row["results_storage_path"], "results_storage_path must be set before Phase 2"

    # Snapshot Phase 1 contents for non-destructive merge assertion.
    phase1_results = json.loads(
        sb.storage.from_("results").download(row["results_storage_path"]).decode("utf-8")
    )
    phase1_rallies = phase1_results["rallies"]
    phase1_shuttle = phase1_results["shuttle_positions"]

    phase1_clip_count = len(
        (
            sb.table("rally_clips")
            .select("id")
            .eq("video_id", args.video_id)
            .execute()
            .data
        )
        or []
    )

    supabase_url = os.environ["SUPABASE_URL"]
    service_role_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    res = requests.post(
        f"{supabase_url}/functions/v1/start-analytics",
        headers={
            "Authorization": f"Bearer {service_role_key}",
            "apikey": service_role_key,
            "Content-Type": "application/json",
        },
        json={"videoId": args.video_id},
    )
    assert res.status_code == 202, f"Edge Function failed: {res.status_code} {res.text}"

    deadline = time.time() + args.timeout_s
    row = None
    while time.time() < deadline:
        row = (
            sb.table("videos")
            .select("status, error")
            .eq("id", args.video_id)
            .single()
            .execute()
            .data
        )
        if row["status"] in ("completed", "failed_phase2"):
            break
        time.sleep(10)
    else:
        raise AssertionError(f"Phase 2 did not complete within {args.timeout_s}s")

    assert row is not None
    assert row["status"] == "completed", (
        f"Phase 2 failed: status={row['status']} error={row.get('error')}"
    )

    # Re-fetch the merged results path (Phase 2 may rewrite the file in place
    # or under the same path — either way, look it up fresh).
    row2 = (
        sb.table("videos")
        .select("results_storage_path")
        .eq("id", args.video_id)
        .single()
        .execute()
        .data
    )
    assert row2["results_storage_path"], "results_storage_path missing after Phase 2"

    merged = json.loads(
        sb.storage.from_("results").download(row2["results_storage_path"]).decode("utf-8")
    )

    # Phase indicator flipped to 'completed'.
    assert merged.get("phase") == "completed", (
        f"Expected phase=='completed' after Phase 2, got {merged.get('phase')!r}"
    )

    # Phase 1 fields must be byte-identical.
    assert merged["rallies"] == phase1_rallies, "Phase 1 rallies were mutated by Phase 2"
    assert merged["shuttle_positions"] == phase1_shuttle, (
        "Phase 1 shuttle_positions were mutated by Phase 2"
    )

    # Phase 2 additions present.
    assert "analytics" in merged, "Phase 2 did not add 'analytics'"
    assert "skeleton_frames" in merged, "Phase 2 did not add 'skeleton_frames'"

    # rally_clips fan-out must not change in Phase 2 (clips are produced in Phase 1).
    clips_after = (
        sb.table("rally_clips")
        .select("id")
        .eq("video_id", args.video_id)
        .execute()
        .data
    ) or []
    assert len(clips_after) == phase1_clip_count, (
        f"Phase 2 mutated rally_clips: was {phase1_clip_count}, now {len(clips_after)}"
    )

    print(f"OK: Phase 2 verified for {args.video_id}.")


if __name__ == "__main__":
    main()
