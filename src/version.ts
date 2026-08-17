/**
 * App version, derived from package.json at build time.
 *
 * package.json is the single source of truth. Every place that displays a
 * version reads it from here, so the header badge and the login badge can no
 * longer drift apart (they did: the header said v2.1 while login said v2.0).
 *
 * The in-app changelog entries in App.vue are deliberately NOT derived from
 * this: each entry is a historical label for a release that already shipped.
 */

/** Full SemVer string, e.g. "2.1.0". */
export const APP_VERSION = __APP_VERSION__

/** Badge form used in the UI, e.g. "v2.1.0". */
export const APP_VERSION_BADGE = `v${APP_VERSION}`
