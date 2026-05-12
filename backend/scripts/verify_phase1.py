"""
Verify Phase 1: invoke process_video against a fixture video and assert outputs.

Usage:
    python -m backend.scripts.verify_phase1 --video-id <id>

Preconditions:
- The video row exists with status 'uploaded' (or 'failed_phase1' for a retry).
- `manual_court_keypoints` is set on the row.
- Env vars: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.

This script exercises the full Phase 1 path through the `process-video` Edge
Function, then asserts the resulting database row, results JSON shape, and
rally_clips fan-out.
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
        default=900,
        help="Seconds to wait for Phase 1 to terminate (default: 900)",
    )
    args = parser.parse_args()

    import requests  # local import to keep top-level import surface small

    sb = _supabase_client()

    # Sanity: precondition.
    row = (
        sb.table("videos")
        .select("status, manual_court_keypoints")
        .eq("id", args.video_id)
        .single()
        .execute()
        .data
    )
    assert row, f"Video {args.video_id} not found"
    assert row["status"] in ("uploaded", "failed_phase1"), (
        f"Bad starting status: {row['status']} (expected 'uploaded' or 'failed_phase1')"
    )
    assert row["manual_court_keypoints"], "Court keypoints required on the row before Phase 1"

    supabase_url = os.environ["SUPABASE_URL"]
    service_role_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    # Invoke via the Edge Function (so we exercise the full path).
    res = requests.post(
        f"{supabase_url}/functions/v1/process-video",
        headers={
            "Authorization": f"Bearer {service_role_key}",
            "apikey": service_role_key,
            "Content-Type": "application/json",
        },
        json={"videoId": args.video_id},
    )
    assert res.status_code == 202, f"Edge Function failed: {res.status_code} {res.text}"

    # Poll for terminal state.
    deadline = time.time() + args.timeout_s
    row = None
    while time.time() < deadline:
        row = (
            sb.table("videos")
            .select("status, results_storage_path, error")
            .eq("id", args.video_id)
            .single()
            .execute()
            .data
        )
        if row["status"] in ("phase1_complete", "failed_phase1"):
            break
        time.sleep(5)
    else:
        raise AssertionError(f"Phase 1 did not complete within {args.timeout_s}s")

    assert row is not None
    assert row["status"] == "phase1_complete", (
        f"Phase 1 failed: status={row['status']} error={row.get('error')}"
    )
    assert row["results_storage_path"], "results_storage_path not set after phase1_complete"

    # Assert results JSON shape.
    results = json.loads(
        sb.storage.from_("results").download(row["results_storage_path"]).decode("utf-8")
    )
    assert results.get("phase") == "phase1", (
        f"Expected phase=='phase1', got {results.get('phase')!r}"
    )
    assert "rallies" in results and isinstance(results["rallies"], list), (
        "Phase 1 results missing 'rallies' list"
    )
    assert "shuttle_positions" in results, "Phase 1 results missing 'shuttle_positions'"
    assert "fps" in results, "Phase 1 results missing 'fps'"
    assert "total_frames" in results, "Phase 1 results missing 'total_frames'"
    assert "skeleton_frames" not in results, "Phase 1 must NOT include skeleton_frames"
    assert "analytics" not in results, "Phase 1 must NOT include analytics"

    # Assert rally_clips rows present.
    clips = (
        sb.table("rally_clips")
        .select("id")
        .eq("video_id", args.video_id)
        .execute()
        .data
    ) or []
    assert len(clips) > 0, "rally_clips rows missing after Phase 1"

    print(
        f"OK: Phase 1 verified for {args.video_id}. "
        f"{len(results['rallies'])} rallies, {len(clips)} clips."
    )


if __name__ == "__main__":
    main()
