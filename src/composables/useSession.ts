import { ref, computed } from "vue";
import type { Session, User } from "@supabase/supabase-js";
import { supabase } from "@/lib/supabase";

const session = ref<Session | null>(null);
const ready = ref(false);

let initialized = false;
function init() {
  if (initialized) return;
  initialized = true;
  supabase.auth.getSession().then(({ data }) => {
    session.value = data.session;
    ready.value = true;
  });
  supabase.auth.onAuthStateChange((_evt, s) => {
    session.value = s;
  });
}

export function useSession() {
  init();
  const user = computed<User | null>(() => session.value?.user ?? null);
  const isAuthenticated = computed(() => !!session.value);
  return { session, user, isAuthenticated, ready };
}

export async function signOut() {
  await supabase.auth.signOut();
}
