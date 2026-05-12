import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders } from "../_shared/cors.ts";
import { signBody } from "../_shared/hmac.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const MODAL_PROCESS_ANALYTICS_URL = Deno.env.get("MODAL_PROCESS_ANALYTICS_URL")!;
const MODAL_SHARED_SECRET = Deno.env.get("MODAL_SHARED_SECRET")!;

if (!MODAL_SHARED_SECRET) {
  throw new Error("MODAL_SHARED_SECRET environment variable is not set");
}

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });

  const auth = req.headers.get("Authorization") ?? "";
  const m = auth.match(/^\s*Bearer\s+(.+?)\s*$/i);
  const jwt = m?.[1];
  if (!jwt) return resp(401, { error: "Missing Authorization" });

  const adminClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
  const { data: { user }, error: authErr } = await adminClient.auth.getUser(jwt);
  if (authErr || !user) return resp(401, { error: "Invalid JWT" });

  const { video_id } = await req.json();
  if (!video_id) return resp(400, { error: "video_id required" });

  // 1. Ownership: user-scoped client enforces RLS so only the owner sees the row.
  const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${jwt}` } },
  });
  const { data: video, error: vErr } = await userClient
    .from("videos").select("*").eq("id", video_id).single();
  if (vErr || !video) return resp(404, { error: "Video not found" });

  // 2. State check.
  if (!["phase1_complete", "failed_phase2"].includes(video.status)) {
    return resp(409, {
      error: `Video is not eligible for analytics (current: ${video.status})`,
    });
  }

  // 3. Preconditions.
  if (!video.manual_court_keypoints) {
    return resp(400, { error: "Court setup required before analytics" });
  }
  if (!video.results_storage_path) {
    return resp(400, { error: "Phase 1 results missing; cannot start analytics" });
  }

  // 4. Status flip (BEFORE Modal call) — deliberate divergence from
  // `process-video`, which flips AFTER Modal returns OK. Phase 2 is
  // user-triggered (the "Continue with full analytics" button), so the
  // flip prevents double-click races. Modal failures roll back to
  // `failed_phase2` below. Orphan risk: if the function dies between
  // flip and Modal call, the row stays in `processing_phase2` with no
  // worker. Acceptable today; a timeout-based watchdog can be added if
  // we observe stuck rows in practice.
  const { error: flipErr } = await adminClient.from("videos")
    .update({
      status: "processing_phase2",
      error: null,
      progress: 0,
      processing_started_at: new Date().toISOString(),
    })
    .eq("id", video_id);
  if (flipErr) {
    console.error("Failed to flip status to processing_phase2", { video_id, error: flipErr });
    return resp(500, { error: "Could not update video status" });
  }

  // 5. Modal call.
  const body = JSON.stringify({ video_id });
  const signature = await signBody(body, MODAL_SHARED_SECRET);

  let modalRes: Response;
  try {
    modalRes = await fetch(MODAL_PROCESS_ANALYTICS_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Signature": signature },
      body,
    });
  } catch (e) {
    // Network-level failure: roll back so the user can retry.
    const msg = e instanceof Error ? e.message : String(e);
    await adminClient.from("videos")
      .update({ status: "failed_phase2", error: `Modal invoke network error: ${msg}` })
      .eq("id", video_id);
    return resp(502, { error: "Modal invoke network error", detail: msg });
  }

  if (!modalRes.ok) {
    const detail = await modalRes.text();
    await adminClient.from("videos")
      .update({
        status: "failed_phase2",
        error: `Modal invoke failed: ${modalRes.status} ${detail}`.slice(0, 1000),
      })
      .eq("id", video_id);
    return resp(502, { error: "Modal rejected", detail });
  }

  return resp(200, { ok: true, videoId: video_id, status: "processing_phase2" });
});

function resp(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}
