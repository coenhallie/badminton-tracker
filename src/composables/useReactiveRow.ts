import { ref, watchEffect, type Ref } from "vue";
import type { RealtimeChannel } from "@supabase/supabase-js";
import { supabase } from "@/lib/supabase";

// Per-call counter ensures unique Supabase channel names so multiple
// consumers of the same (table, id) don't collide. Reusing a channel name
// returns the existing channel; calling .on() on a channel that has already
// .subscribe()d throws "cannot add postgres_changes callbacks after subscribe()".
let channelSeq = 0;

export function useReactiveRow<T>(table: string, id: Ref<string | null | undefined>) {
  const row = ref<T | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  // Synchronous effect on purpose. The previous version was `async` and called
  // onCleanup AFTER an await; Vue only reliably attaches a cleanup registered
  // before the first await, so a rapid `id` change could leak a channel.
  // Awaiting inside load() keeps both dependency tracking and cleanup
  // registration synchronous.
  watchEffect((onCleanup) => {
    const rowId = id.value;

    let disposed = false;
    let channel: RealtimeChannel | null = null;
    let onVisible: (() => void) | null = null;
    onCleanup(() => {
      disposed = true;
      if (channel) supabase.removeChannel(channel);
      if (onVisible) document.removeEventListener("visibilitychange", onVisible);
    });

    if (!rowId) {
      row.value = null;
      return;
    }

    // Monotonic write ordering. A re-fetch and a realtime push can be in flight
    // at the same time, and a SELECT issued earlier may resolve later — applying
    // it would overwrite newer data with older. Every write takes a ticket, and
    // only a ticket at least as new as the last applied one wins.
    let ticket = 0;
    let applied = 0;
    const apply = (value: T | null, t: number) => {
      if (disposed || t < applied) return;
      applied = t;
      row.value = value;
    };

    async function load(quiet: boolean) {
      const t = ++ticket;
      if (!quiet) {
        loading.value = true;
        error.value = null;
      }
      const { data, error: e } = await supabase
        .from(table)
        .select("*")
        .eq("id", rowId)
        .maybeSingle();
      if (disposed) return;
      if (e) {
        // A failed BACKGROUND refresh must not replace good data with an error
        // banner — the next rejoin or visibility change retries. Only the
        // initial, user-visible load reports.
        if (!quiet) {
          error.value = e.message;
          loading.value = false;
        }
        return;
      }
      apply(data as T | null, t);
      if (!quiet) loading.value = false;
    }

    // Eager first read: first paint must not wait on the websocket handshake,
    // and the row still renders if realtime is unavailable entirely.
    void load(false);

    channel = supabase
      .channel(`${table}-row-${rowId}-${++channelSeq}`)
      .on(
        "postgres_changes",
        { event: "UPDATE", schema: "public", table, filter: `id=eq.${rowId}` },
        (payload) => apply(payload.new as T, ++ticket),
      )
      .on(
        "postgres_changes",
        { event: "DELETE", schema: "public", table, filter: `id=eq.${rowId}` },
        () => apply(null, ++ticket),
      )
      .subscribe((status) => {
        // Fires on the initial join AND on every automatic rejoin after the
        // socket drops. Re-reading here is what closes the two gaps that could
        // otherwise strand a caller on a finished job:
        //   1. the mount race — the eager load() above resolves BEFORE the
        //      subscription exists, so an UPDATE in that window was lost;
        //   2. the reconnect gap — every UPDATE delivered while disconnected is
        //      gone for good, and nothing re-read the row afterwards. This is
        //      how AnalysisProgress could sit at 40% on a completed analysis.
        if (status === "SUBSCRIBED") void load(true);
      });

    // Background tabs get their timers throttled, so the realtime heartbeat can
    // lapse and the socket close without a prompt rejoin. AnalysisProgress
    // explicitly invites the user to leave the tab ("You can leave this tab;
    // results will be saved"), so make coming back to it a refresh. Covers the
    // case where the rejoin above is slow or never happens.
    onVisible = () => {
      if (document.visibilityState === "visible") void load(true);
    };
    document.addEventListener("visibilitychange", onVisible);
  });

  return { row, loading, error };
}
