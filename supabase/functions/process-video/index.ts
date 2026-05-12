import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders } from "../_shared/cors.ts";
import { signBody } from "../_shared/hmac.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const MODAL_PROCESS_URL = Deno.env.get("MODAL_PROCESS_URL")!;
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

  const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${jwt}` } },
  });
  const { data: video, error: vErr } = await userClient
    .from("videos").select("*").eq("id", video_id).single();
  if (vErr || !video) return resp(404, { error: "Video not found" });

  if (video.status !== "uploaded") {
    return resp(409, { error: `Video is not in 'uploaded' state (current: ${video.status})` });
  }
  if (!video.manual_court_keypoints) {
    return resp(400, { error: "Court setup required before processing" });
  }

  const { data: signed, error: sErr } = await adminClient
    .storage.from("videos").createSignedUrl(video.storage_path, 3600);
  if (sErr || !signed) return resp(500, { error: "Could not sign video URL" });

  // Flip status BEFORE invoking Modal. Mirrors start-analytics. Prevents a
  // double-click race where a second call slips past the `status==='uploaded'`
  // guard while the first Modal invocation is in flight. If Modal then
  // rejects, we roll back below.
  const { error: flipErr } = await adminClient.from("videos")
    .update({ status: "processing_phase1", processing_started_at: new Date().toISOString() })
    .eq("id", video_id);
  if (flipErr) {
    console.error("Failed to flip status to processing_phase1", { video_id, error: flipErr });
    return resp(500, { error: "Could not update video status" });
  }

  const body = JSON.stringify({
    video_id,
    owner_id: user.id,
    video_url: signed.signedUrl,
  });
  const signature = await signBody(body, MODAL_SHARED_SECRET);

  const modalRes = await fetch(MODAL_PROCESS_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Signature": signature },
    body,
  });
  if (!modalRes.ok) {
    const text = await modalRes.text();
    // Roll back so the user can retry.
    await adminClient.from("videos")
      .update({ status: "failed_phase1", error: `Modal rejected: ${text.slice(0, 1000)}` })
      .eq("id", video_id);
    return resp(502, { error: "Modal rejected", detail: text });
  }

  return resp(200, { ok: true });
});

function resp(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}
