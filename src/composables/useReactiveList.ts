import { ref, watchEffect, type Ref } from "vue";
import { supabase } from "@/lib/supabase";

interface ListFilter { column: string; value: string; }
interface Options { orderBy?: string; ascending?: boolean; }

export function useReactiveList<T extends { id: string | number }>(
  table: string,
  filter: Ref<ListFilter | null>,
  options: Options = {}
) {
  const items = ref<T[]>([]) as Ref<T[]>;
  const loading = ref(false);
  const error = ref<string | null>(null);

  watchEffect(async (onCleanup) => {
    if (!filter.value) {
      items.value = [];
      return;
    }
    loading.value = true;
    error.value = null;
    const { column, value } = filter.value;

    let q = supabase.from(table).select("*").eq(column, value);
    if (options.orderBy) {
      q = q.order(options.orderBy, { ascending: options.ascending ?? true });
    }
    const { data, error: e } = await q;
    if (e) {
      error.value = e.message;
      loading.value = false;
      return;
    }
    items.value = (data ?? []) as T[];
    loading.value = false;

    const channel = supabase
      .channel(`${table}-list-${column}-${value}`)
      .on("postgres_changes",
          { event: "INSERT", schema: "public", table, filter: `${column}=eq.${value}` },
          (p) => { items.value = [...items.value, p.new as T]; })
      .on("postgres_changes",
          { event: "DELETE", schema: "public", table, filter: `${column}=eq.${value}` },
          (p) => { items.value = items.value.filter((it) => it.id !== (p.old as T).id); })
      .subscribe();

    onCleanup(() => { supabase.removeChannel(channel); });
  });

  return { items, loading, error };
}
