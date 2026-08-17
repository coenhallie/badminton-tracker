/// <reference types="vite/client" />

/**
 * Build info derived from git at build time and injected by vite.config.ts.
 * The shape is defined once, in scripts/git-version.ts.
 */
declare const __APP_BUILD__: import('./scripts/git-version').BuildInfo
