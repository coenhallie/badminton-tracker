<script setup lang="ts">
import { ref, computed } from "vue";
import { supabase } from "@/lib/supabase";
import { useTheme } from "@/composables/useTheme";

const emit = defineEmits<{
  "show-changelog": [];
}>();

const { isDark, toggleTheme } = useTheme();

const email = ref("");
const password = ref("");
const error = ref<string | null>(null);
const submittingEmail = ref(false);

const busy = computed(() => submittingEmail.value);

async function signInEmail() {
  if (busy.value) return;
  submittingEmail.value = true;
  error.value = null;
  const { error: e } = await supabase.auth.signInWithPassword({
    email: email.value,
    password: password.value,
  });
  submittingEmail.value = false;
  if (e) error.value = e.message;
}
</script>

<template>
  <div class="login-page">
    <button
      class="theme-toggle"
      @click="toggleTheme"
      :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
      type="button"
    >
      <svg v-if="isDark" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="5" />
        <line x1="12" y1="1" x2="12" y2="3" />
        <line x1="12" y1="21" x2="12" y2="23" />
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
        <line x1="1" y1="12" x2="3" y2="12" />
        <line x1="21" y1="12" x2="23" y2="12" />
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
      </svg>
      <svg v-else xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      </svg>
    </button>

    <main class="login-shell">
      <div class="brand">
        <h1 class="wordmark">SHUTTL.</h1>
        <button
          class="alpha-badge"
          type="button"
          @click="emit('show-changelog')"
        >
          beta v2.0
        </button>
      </div>
      <p class="subtitle">Sign in to continue</p>

      <div class="card">
        <form class="form" @submit.prevent="signInEmail">
          <div class="field">
            <label class="field-label" for="login-email">Email</label>
            <input
              id="login-email"
              v-model="email"
              class="field-input"
              type="email"
              autocomplete="email"
              :disabled="busy"
              required
            />
          </div>
          <div class="field">
            <label class="field-label" for="login-password">Password</label>
            <input
              id="login-password"
              v-model="password"
              class="field-input"
              type="password"
              autocomplete="current-password"
              :disabled="busy"
              required
            />
          </div>

          <button
            class="primary-btn"
            type="submit"
            :disabled="busy"
          >
            <span v-if="submittingEmail" class="spinner" />
            {{ submittingEmail ? "Signing in…" : "Sign in" }}
          </button>
        </form>

        <p v-if="error" class="error-banner" role="alert">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{{ error }}</span>
        </p>
      </div>

      <p class="hint">
        Registration is closed. Contact the admin if you need an account.
      </p>
    </main>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px 16px;
  background: var(--color-bg);
  color: var(--color-text);
}

.theme-toggle {
  position: fixed;
  top: 16px;
  right: 24px;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  border: 1px solid var(--color-border);
  background: var(--color-bg-secondary);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all 0.2s;
  z-index: 10;
}

.theme-toggle:hover {
  background: var(--color-bg-hover);
  color: var(--color-text);
  border-color: var(--color-border-hover);
}

.login-shell {
  width: 100%;
  max-width: 400px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
}

.wordmark {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-text-heading);
  letter-spacing: -0.01em;
}

.alpha-badge {
  background: linear-gradient(135deg, var(--color-accent) 0%, var(--color-accent-dark) 100%);
  color: #000;
  font-size: 0.65rem;
  font-weight: 600;
  padding: 3px 8px;
  border: none;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  transition: all 0.2s ease;
  margin-left: 4px;
}

.alpha-badge:hover {
  background: linear-gradient(135deg, var(--color-accent-hover) 0%, var(--color-accent) 100%);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(34, 197, 94, 0.3);
}

.subtitle {
  margin-top: 8px;
  margin-bottom: 24px;
  font-size: 0.875rem;
  color: var(--color-text-secondary);
}

.card {
  width: 100%;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  padding: 32px;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field-label {
  color: var(--color-text-secondary);
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.field-input {
  background: var(--color-bg-input);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  padding: 10px 12px;
  font-size: 0.9rem;
  color: var(--color-text);
  font-family: inherit;
  transition: border-color 0.2s ease;
  width: 100%;
}

.field-input:focus {
  outline: none;
  border-color: var(--color-accent);
}

.field-input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.primary-btn {
  margin-top: 8px;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 24px;
  background: var(--color-accent);
  border: none;
  border-radius: 0;
  color: #000;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s ease, opacity 0.2s ease;
  font-family: inherit;
}

.primary-btn:hover:not(:disabled) {
  background: var(--color-accent-dark);
}

.primary-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.error-banner {
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background: rgba(239, 68, 68, 0.08);
  border-left: 2px solid var(--color-error);
  color: var(--color-error);
  font-size: 0.85rem;
  line-height: 1.4;
}

.error-banner svg {
  flex-shrink: 0;
}

.spinner {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(0, 0, 0, 0.2);
  border-top-color: #000;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.hint {
  margin-top: 24px;
  font-size: 0.8rem;
  color: var(--color-text-tertiary);
  text-align: center;
  max-width: 320px;
  line-height: 1.5;
}

@media (max-width: 480px) {
  .card {
    padding: 24px;
  }
  .wordmark {
    font-size: 1.25rem;
  }
  .theme-toggle {
    top: 12px;
    right: 12px;
  }
}
</style>
