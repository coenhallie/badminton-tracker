/**
 * Derives the app version from git at build time.
 *
 * SHUTTL is a continuously-deployed app, not a published library, so the version
 * is not hand-written anywhere. It answers "which commit is running?", not "is
 * this a breaking change?". That means it comes from git:
 *
 *   git describe --tags --long   ->  v2.1.0-24-gd90533a
 *                                        |     |      |
 *                          last release tag   |      commit sha
 *                                    commits since that tag
 *
 * which is normalized to a SemVer string using the build-metadata field:
 *
 *   2.1.0+24.gd90533a
 *
 * SemVer reserves everything after `+` for exactly this and excludes it from
 * version precedence, so the string stays spec-compliant and machine-parseable.
 *
 * See docs/VERSIONING.md.
 */

import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

// Pin git to the repo root so the result does not depend on where vite was invoked.
const repoRoot = fileURLToPath(new URL('..', import.meta.url))

function git(args: string[]): string {
  return execFileSync('git', args, {
    cwd: repoRoot,
    encoding: 'utf-8',
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim()
}

export interface BuildInfo {
  /** Last release tag without the `v`, e.g. "2.1.0". Null if no tag was reachable. */
  release: string | null
  /** Commits between that tag and the build. Null if no tag was reachable. */
  commitsSinceRelease: number | null
  /** Short commit sha of the build, e.g. "d90533a". */
  commit: string | null
  /** ISO committer date of that commit. */
  commitDate: string | null
  /** Whether the working tree had uncommitted changes at build time. */
  dirty: boolean
  /** Full SemVer string with build metadata, e.g. "2.1.0+24.gd90533a". */
  version: string
  /** Where the version came from, for diagnosing a degraded build. */
  source: 'git' | 'git-untagged' | 'package.json'
  /** ISO timestamp of the build itself. */
  builtAt: string
}

/**
 * @param packageVersion - package.json "version", used as the fallback release
 *   number when git has no tag to anchor to.
 */
export function resolveBuildInfo(packageVersion: string): BuildInfo {
  const builtAt = new Date().toISOString()

  let describe: string
  let commitDate: string
  try {
    // --long always emits the -<n>-g<sha> suffix, even exactly on a tag.
    // --always degrades to a bare sha when no tag is reachable (shallow clones).
    describe = git(['describe', '--tags', '--long', '--dirty', '--always'])
    commitDate = git(['log', '-1', '--format=%cI'])
  } catch {
    // No git binary, or not a repo (e.g. building from a source tarball).
    return {
      release: packageVersion,
      commitsSinceRelease: null,
      commit: null,
      commitDate: null,
      dirty: false,
      version: packageVersion,
      source: 'package.json',
      builtAt,
    }
  }

  let dirty = false
  if (describe.endsWith('-dirty')) {
    dirty = true
    describe = describe.slice(0, -'-dirty'.length)
  }

  const tagged = /^v?(.+)-(\d+)-g([0-9a-f]+)$/.exec(describe)
  const release = tagged?.[1]
  const commitsRaw = tagged?.[2]
  const commit = tagged?.[3]

  // No reachable tag: keep the sha and date, which are still useful, but we have
  // no release to anchor to. Happens on CI shallow clones (fix with fetch-depth: 0).
  if (!release || !commitsRaw || !commit) {
    return {
      release: null,
      commitsSinceRelease: null,
      commit: describe,
      commitDate,
      dirty,
      version: `0.0.0+g${describe}${dirty ? '.dirty' : ''}`,
      source: 'git-untagged',
      builtAt,
    }
  }

  const commitsSinceRelease = Number(commitsRaw)

  if (release !== packageVersion) {
    console.warn(
      `[version] package.json says ${packageVersion} but the latest git tag is v${release}. ` +
        `Bump package.json and tag in the same commit - see docs/VERSIONING.md.`,
    )
  }

  const metadata = [
    commitsSinceRelease > 0 ? String(commitsSinceRelease) : null,
    commitsSinceRelease > 0 || dirty ? `g${commit}` : null,
    dirty ? 'dirty' : null,
  ].filter(Boolean)

  return {
    release,
    commitsSinceRelease,
    commit,
    commitDate,
    dirty,
    version: metadata.length ? `${release}+${metadata.join('.')}` : release,
    source: 'git',
    builtAt,
  }
}
