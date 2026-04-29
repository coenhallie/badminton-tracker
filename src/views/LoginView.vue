<script setup lang="ts">
import { ref } from "vue";
import { supabase } from "@/lib/supabase";

const email = ref("");
const password = ref("");
const error = ref<string | null>(null);
const submitting = ref(false);

async function signInEmail() {
  submitting.value = true;
  error.value = null;
  const { error: e } = await supabase.auth.signInWithPassword({
    email: email.value,
    password: password.value,
  });
  submitting.value = false;
  if (e) error.value = e.message;
}

async function signInGoogle() {
  const { error: e } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo: window.location.origin },
  });
  if (e) error.value = e.message;
}
</script>

<template>
  <div class="login">
    <h1>Sign in</h1>
    <form @submit.prevent="signInEmail">
      <input v-model="email" type="email" placeholder="email" autocomplete="email" required />
      <input v-model="password" type="password" placeholder="password" autocomplete="current-password" required />
      <button type="submit" :disabled="submitting">{{ submitting ? "Signing in…" : "Sign in" }}</button>
    </form>
    <button @click="signInGoogle">Sign in with Google</button>
    <p v-if="error" class="error">{{ error }}</p>
    <p class="hint">Registration is closed. Contact the admin if you need an account.</p>
  </div>
</template>

<style scoped>
.login { max-width: 320px; margin: 8rem auto; display: flex; flex-direction: column; gap: 1rem; }
.error { color: tomato; }
.hint { font-size: 0.85rem; opacity: 0.7; }
</style>
