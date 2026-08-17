/**
 * App version, derived from git at build time.
 *
 * Nothing here is hand-maintained. `scripts/git-version.js` runs `git describe`
 * during the build and injects the result as `__APP_BUILD__`; this module only
 * formats it for display. See docs/VERSIONING.md.
 *
 * The in-app changelog entries in App.vue are deliberately NOT derived from this:
 * each entry is a historical label for a release that already shipped.
 */

export const BUILD = __APP_BUILD__

/** Full SemVer string with build metadata, e.g. "2.1.0+24.gd90533a". */
export const APP_VERSION = BUILD.version

/**
 * Compact badge form.
 *
 * Exactly on a release tag it is just "v2.1.0". Past one it carries the commit
 * count, "v2.1.0+24", so it is obvious at a glance that the running build is
 * ahead of the last release rather than being that release.
 */
export const APP_VERSION_BADGE = BUILD.release
  ? `v${BUILD.release}${BUILD.commitsSinceRelease ? `+${BUILD.commitsSinceRelease}` : ''}`
  : `v${BUILD.version}`

function formatDate(iso: string | null): string | null {
  if (!iso) return null
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return null
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

/** Committer date of the build's commit, e.g. "Aug 17, 2026". */
export const APP_COMMIT_DATE = formatDate(BUILD.commitDate)

/** Date the bundle was built, e.g. "Aug 17, 2026". */
export const APP_BUILD_DATE = formatDate(BUILD.builtAt)

/**
 * One-line provenance shown in the changelog modal, e.g.
 * "2.1.0+24.gd90533a - commit d90533a, Aug 17, 2026 - built Aug 17, 2026".
 */
export const APP_BUILD_LINE = [
  APP_VERSION,
  BUILD.commit && APP_COMMIT_DATE
    ? `commit ${BUILD.commit}, ${APP_COMMIT_DATE}`
    : BUILD.commit
      ? `commit ${BUILD.commit}`
      : null,
  APP_BUILD_DATE ? `built ${APP_BUILD_DATE}` : null,
  BUILD.dirty ? 'uncommitted changes' : null,
]
  .filter(Boolean)
  .join(' - ')
