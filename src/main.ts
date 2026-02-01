import { createApp } from 'vue'
import { convexVue } from 'convex-vue'
import App from './App.vue'

const app = createApp(App)

// Initialize Convex (for composables that use Convex directly)
if (import.meta.env.VITE_CONVEX_URL) {
  app.use(convexVue, {
    url: import.meta.env.VITE_CONVEX_URL,
  })
}

app.mount('#app')
