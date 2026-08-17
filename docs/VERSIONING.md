# Versioning

No version number is ever typed by hand.

- **Release numbers** are computed from Conventional Commit messages -
  `npm run release`. See [Cutting a release](#cutting-a-release).
- **Every build in between** derives its version from `git describe` at build
  time. See [How the version is derived](#how-the-version-is-derived).

## Why this shape

There are two versioning problems, and they have different answers:

**Libraries** published for other people to depend on need hand-curated SemVer.
Deciding "is this major or minor?" is a judgement about breaking someone else's
build, and no tool can make it. This is what `npm publish` semantics assume.

**Continuously-deployed applications** need something else. Nobody depends on
SHUTTL's version number, and the question a version has to answer is "which commit
is running right now, and how old is it?". That is a fact about git, so it is
derived from git rather than typed by a person. This is the pattern across
continuously-deployed codebases: a SemVer-shaped core so version parsers keep
working, with the short git SHA carried in the build-metadata field so any build
maps back to the exact sources it came from
([aspect.build](https://aspect.build/blog/versioning-releases-from-a-monorepo),
[GitVersion](https://gitversion.net/docs/reference/version-increments)).

Hand-bumping a field in `package.json` on every deploy is the failure mode this
replaces - it is the same manual step that had the header badge reading `beta v2.1`
while the login screen read `beta v2.0`.

## How the version is derived

```
git describe --tags --long --dirty     ->  v2.1.0-24-gd90533a
                                               |    |      |
                                 last release tag   |     commit sha
                                       commits since that tag

normalized to SemVer                   ->  2.1.0+24.gd90533a
```

Everything after `+` is SemVer's **build metadata** field. The spec reserves it for
exactly this and excludes it from version precedence, so the string stays
spec-compliant and parseable while carrying the git provenance.

`scripts/git-version.js` runs at build time, `vite.config.ts` injects the result as
`__APP_BUILD__`, and `src/version.ts` formats it:

```
git (tags + HEAD)
     |  scripts/git-version.js       git describe, git log -1
     v
vite.config.ts  define: { __APP_BUILD__ }
     v
src/version.ts  APP_VERSION_BADGE / APP_BUILD_LINE
     |
     +--> src/App.vue              header badge + changelog provenance line
     +--> src/views/LoginView.vue  login badge
```

## What gets displayed

| Where | Shows | Example |
|---|---|---|
| Badge, on a release tag | `v<release>` | `v2.1.0` |
| Badge, past a release | `v<release>+<commits>` | `v2.1.0+24` |
| Badge tooltip, and the changelog modal | full provenance | `2.1.0+24.gd90533a - commit d90533a, Aug 17, 2026 - built Aug 17, 2026` |

The badge stays short; the full string, commit, commit date and build date live in
the tooltip and in the changelog modal.

### On dates

Git does not record when a commit was *pushed* - a push is a transfer, not
something stamped into the object. Two dates are recorded and both are shown:

- **commit date** - the committer date of the commit the bundle was built from.
  This is how old the code is, and is the closest honest answer to "when was this
  last pushed".
- **build date** - when the bundle itself was produced. For a deployed app this is
  effectively "how old is this deploy", which is usually the more useful of the two.

## Version format

`MAJOR.MINOR.PATCH[+BUILD]`, tags prefixed with `v`. No `-alpha` / `-beta`
prerelease suffix - the version number plus the git history is the whole story.

| | |
|---|---|
| MAJOR | breaking change to the data model, API, or pipeline contract |
| MINOR | new user-facing capability, backwards compatible |
| PATCH | bug fixes and internal work only |
| BUILD | derived, never typed: `<commits-since-tag>.g<sha>`, plus `.dirty` if the tree was not clean |

The tag is the only thing a human chooses, and only when cutting a release.

## Cutting a release

The release number is **computed from commit messages**, not chosen. That is the
job of [`commit-and-tag-version`](https://github.com/absolute-version/commit-and-tag-version):

```sh
npm run release:dry   # preview: what version, and what would go in the changelog
npm run release       # bump package.json, write CHANGELOG.md, commit, tag
git push origin <branch> --follow-tags
```

`npm run release` reads every commit since the last tag and maps
[Conventional Commit](https://www.conventionalcommits.org/) prefixes to a bump:

| Commit | Bump |
|---|---|
| `fix: …` | PATCH |
| `feat: …` | MINOR |
| `feat!: …` or a `BREAKING CHANGE:` footer | MAJOR |

Because it bumps `package.json` and creates the tag in the same commit, the
fallback version and the tag can never disagree. The build warns if they drift.

This is the reason commit messages are worth writing carefully: the version and
the changelog are both derived from them. Two early commits
(`rally seperation optimizations`, `remove rally only option`) predate the
convention and are simply skipped.

It runs entirely locally - no CI, no npm publish, no remote writes. Nothing leaves
the machine until you push, so a release can be inspected and thrown away with
`git reset --hard HEAD~1 && git tag -d vX.Y.Z`.

Between releases nothing needs doing: every build past `v2.2.0` reports itself as
`2.2.0+<n>.g<sha>` automatically.

Useful afterwards:

```sh
git describe --tags      # which release HEAD is built on top of
git log v2.1.0..HEAD     # what is unreleased right now
git show v2.1.0          # what a given release was
```

## When git is not available

`scripts/git-version.js` degrades in two steps rather than failing the build. The
`source` field on `__APP_BUILD__` records which path was taken, so a degraded build
is diagnosable rather than silently wrong:

| `source` | When | Version |
|---|---|---|
| `git` | normal | `2.1.0+24.gd90533a` |
| `git-untagged` | repo present but no tag reachable - typically a CI shallow clone | `0.0.0+gd90533a` (sha and date survive) |
| `package.json` | no git binary, or not a repo - e.g. building from a source tarball | `2.1.0` |

There is no CI in this repo today; builds run locally with full history, so the
`git` path is the one in use. If CI is added later it must fetch tags and full
history, or every build will report `git-untagged`:

```yaml
# GitHub Actions
- uses: actions/checkout@v4
  with:
    fetch-depth: 0    # default is a depth-1 clone with no tags
```

## Two changelogs, two audiences

There are deliberately two, and they are not redundant:

| | `CHANGELOG.md` | the changelog in `src/App.vue` |
|---|---|---|
| Audience | developers | end users |
| Content | every commit, grouped by type | prose describing what changed for a player |
| Written by | generated by `npm run release` | by hand, when cutting a release |
| Edit it? | **no** - regenerated each release | yes, that is the point |

The 12 in-app entries are **not** derived from git. Each is a historical label for
a release that already shipped. Deriving them would relabel old entries on every
bump, and a generated commit log is not the same artifact as a changelog someone
reads to find out what the app does now.

Adding the in-app entry is the one genuinely manual step left in a release, and it
is manual because writing for users is not something a commit log can do.

## Historical tags

Tags for v1.0.0 through v2.1.0 were backfilled from the in-app changelog after the
fact. They were placed on the last commit at or before each entry's date; where two
entries shared a date, commit content broke the tie. Each tag's annotation records
how it was anchored, so an inferred tag is never mistaken for a certain one:

- **exact** - an explicit version-bump commit exists (v1.8.0, v2.0.0), or the commit
  was the published tip at the time (v2.1.0)
- **content** - the commit subject matches the changelog bullets (v1.3.0, v1.4.0,
  v1.5.0)
- **date** - nearest commit on or before the release date, nothing more specific
  (v1.0.0, v1.1.0, v1.6.0, v1.7.0, v1.9.0)

The early history is coarser than the changelog, so several entries describe work
with no distinct commit behind it. v1.6.0 is a partial case: its commit covers the
rally-segmentation bullet, but nothing in history corresponds to its headline
TrackNetV3 bullet, so it is anchored by date rather than content.

**v1.2.0 has no tag at all** - the extreme case of the same problem. Its entry is
dated February 7, 2026 and describes skeleton/minicourt smoothing work, but the
repository contains no commits whatsoever between February 1 and February 25, 2026.
There is no commit that release could point at. Tagging the nearest earlier commit
would have claimed that work shipped in a commit written before it existed, so the
version was left untagged rather than anchored to something false.
