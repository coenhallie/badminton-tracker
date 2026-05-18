import { ref, watchEffect, type Ref } from "vue";
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

  watchEffect(async (onCleanup) => {
    if (!id.value) {
      row.value = null;
      return;
    }
    loading.value = true;
    error.value = null;

    const { data, error: e } = await supabase.from(table).select("*").eq("id", id.value).maybeSingle();
    if (e) {
      error.value = e.message;
      loading.value = false;
      return;
    }
    row.value = data as T | null;
    loading.value = false;

    const channel = supabase
      .channel(`${table}-row-${id.value}-${++channelSeq}`)
      .on("postgres_changes",
          { event: "UPDATE", schema: "public", table, filter: `id=eq.${id.value}` },
          (payload) => { row.value = payload.new as T; })
      .on("postgres_changes",
          { event: "DELETE", schema: "public", table, filter: `id=eq.${id.value}` },
          () => { row.value = null; })
      .subscribe();

    onCleanup(() => { supabase.removeChannel(channel); });
  });

  return { row, loading, error };
}
