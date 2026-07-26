import { ref, watchEffect, type Ref } from "vue";
import type { RealtimeChannel } from "@supabase/supabase-js";
import { supabase } from "@/lib/supabase";

interface ListFilter { column: string; value: string; }
interface Options { orderBy?: string; ascending?: boolean; }

// Per-call counter — see comment in useReactiveRow.ts. Same hazard applies:
// reusing a channel name returns the existing channel and .on() after
// .subscribe() throws.
let channelSeq = 0;

export function useReactiveList<T extends { id: string | number }>(
  table: string,
  filter: Ref<ListFilter | null>,
  options: Options = {}
) {
  const items = ref<T[]>([]) as Ref<T[]>;
  const loading = ref(false);
  const error = ref<string | null>(null);

  // Synchronous effect, cleanup registered before any await — see the note in
  // useReactiveRow.ts for why the async version was a channel-leak hazard.
  watchEffect((onCleanup) => {
    const f = filter.value;

    let disposed = false;
    let channel: RealtimeChannel | null = null;
    let onVisible: (() => void) | null = null;
    onCleanup(() => {
      disposed = true;
      if (channel) supabase.removeChannel(channel);
      if (onVisible) document.removeEventListener("visibilitychange", onVisible);
    });

    if (!f) {
      items.value = [];
      return;
    }
    const { column, value } = f;

    const sort = (rows: T[]): T[] => {
      const key = options.orderBy;
      if (!key) return rows;
      const dir = (options.ascending ?? true) ? 1 : -1;
      return [...rows].sort((a, b) => {
        const av = (a as Record<string, unknown>)[key] as string;
        const bv = (b as Record<string, unknown>)[key] as string;
        if (av === bv) return 0;
        return (av < bv ? -1 : 1) * dir;
      });
    };

    async function load(quiet: boolean) {
      if (!quiet) {
        loading.value = true;
        error.value = null;
      }
      let q = supabase.from(table).select("*").eq(column, value);
      if (options.orderBy) {
        q = q.order(options.orderBy, { ascending: options.ascending ?? true });
      }
      const { data, error: e } = await q;
      if (disposed) return;
      if (e) {
        // Background refreshes stay silent — see useReactiveRow.load().
        if (!quiet) {
          error.value = e.message;
          loading.value = false;
        }
        return;
      }
      const fetched = (data ?? []) as T[];
      // Merge rather than replace. A realtime INSERT can land while this SELECT
      // is in flight, after the query's snapshot was taken — a wholesale replace
      // would silently drop that row, which is the exact class of quiet data
      // loss this change exists to remove.
      const fetchedIds = new Set(fetched.map((r) => r.id));
      const localOnly = items.value.filter((r) => !fetchedIds.has(r.id));
      items.value = localOnly.length ? sort([...fetched, ...localOnly]) : fetched;
      if (!quiet) loading.value = false;
    }

    void load(false);

    channel = supabase
      .channel(`${table}-list-${column}-${value}-${++channelSeq}`)
      .on(
        "postgres_changes",
        { event: "INSERT", schema: "public", table, filter: `${column}=eq.${value}` },
        (p) => {
          const incoming = p.new as T;
          // Dedupe by id: a re-fetch on rejoin and this push can both deliver
          // the same row, and appending blindly would double it in the list.
          if (items.value.some((it) => it.id === incoming.id)) return;
          items.value = [...items.value, incoming];
        },
      )
      .on(
        "postgres_changes",
        { event: "DELETE", schema: "public", table, filter: `${column}=eq.${value}` },
        (p) => {
          items.value = items.value.filter((it) => it.id !== (p.old as T).id);
        },
      )
      .subscribe((status) => {
        // Initial join and every automatic rejoin. Recovers rows inserted during
        // the mount race (before the subscription existed) and every row missed
        // while the socket was down — for processing_logs that is a silently
        // truncated log stream with no gap marker to hint at it.
        if (status === "SUBSCRIBED") void load(true);
      });

    onVisible = () => {
      if (document.visibilityState === "visible") void load(true);
    };
    document.addEventListener("visibilitychange", onVisible);
  });

  return { items, loading, error };
}
