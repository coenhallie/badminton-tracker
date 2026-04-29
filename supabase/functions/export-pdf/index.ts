import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders } from "../_shared/cors.ts";
import { signBody } from "../_shared/hmac.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const MODAL_PDF_URL = Deno.env.get("MODAL_PDF_URL")!;
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

  const reqBody = await req.json();
  const { video_id, config } = reqBody;
  if (!video_id) return resp(400, { error: "video_id required" });

  const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${jwt}` } },
  });
  const { data: video, error: vErr } = await userClient
    .from("videos").select("results_storage_path, storage_path")
    .eq("id", video_id).single();
  if (vErr || !video) return resp(404, { error: "Video not found" });

  const body = JSON.stringify({
    video_id,
    owner_id: user.id,
    results_storage_path: video.results_storage_path,
    video_storage_path: video.storage_path,
    config,
  });
  const signature = await signBody(body, MODAL_SHARED_SECRET);

  const modalRes = await fetch(MODAL_PDF_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Signature": signature },
    body,
  });
  if (!modalRes.ok) {
    return resp(502, { error: "Modal PDF call failed", detail: await modalRes.text() });
  }
  return resp(200, await modalRes.json());
});

function resp(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}
