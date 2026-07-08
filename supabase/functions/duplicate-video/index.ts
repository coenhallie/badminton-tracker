import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders } from "../_shared/cors.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

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

  // RLS-scoped read: only the owner sees the row.
  const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${jwt}` } },
  });
  const { data: video, error: vErr } = await userClient
    .from("videos").select("*").eq("id", video_id).single();
  if (vErr || !video) return resp(404, { error: "Video not found" });

  if (["processing_phase1", "processing_phase2", "processing"].includes(video.status)) {
    return resp(409, { error: `Source video is mid-processing (${video.status})` });
  }
  if (!video.manual_court_keypoints) {
    return resp(400, { error: "Source video has no court keypoints to copy" });
  }

  const newId = crypto.randomUUID();
  const newPath = `${user.id}/${newId}.mp4`;
  const newVariant = video.pipeline_variant === "gb_fusion" ? "legacy" : "gb_fusion";

  // Server-side copy: identical bytes, no re-upload.
  const { error: copyErr } = await adminClient.storage
    .from("videos").copy(video.storage_path, newPath);
  if (copyErr) return resp(500, { error: `Storage copy failed: ${copyErr.message}` });

  const { error: insErr } = await adminClient.from("videos").insert({
    id: newId,
    owner_id: user.id,
    filename: video.filename,
    size: video.size,
    storage_path: newPath,
    status: "uploaded",
    manual_court_keypoints: video.manual_court_keypoints,
    player_labels: video.player_labels,
    pipeline_variant: newVariant,
    source_video_id: video.id,
  });
  if (insErr) {
    // Best-effort cleanup of the copied object so a retry doesn't collide.
    try {
      await adminClient.storage.from("videos").remove([newPath]);
    } catch (_) { /* ignore */ }
    return resp(500, { error: `Row insert failed: ${insErr.message}` });
  }

  return resp(200, { new_video_id: newId, pipeline_variant: newVariant });
});

function resp(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}
