# Versioning

The app follows [Semantic Versioning](https://semver.org/) and marks every release
with an annotated git tag, so a version always resolves to an exact commit and a
commit always resolves to the release it shipped in.

## Source of truth

`package.json` `"version"` is the **only** place the current version is written.

```
package.json  "version": "2.1.0"
     |
     |  vite.config.ts  define: { __APP_VERSION__ }
     v
src/version.ts  APP_VERSION / APP_VERSION_BADGE
     |
     +--> src/App.vue          header badge
     +--> src/views/LoginView.vue  login badge
```

Nothing else hardcodes a version string. Before this convention the header read
`beta v2.1` while the login screen read `beta v2.0` - the two had drifted a full
release apart because both were typed by hand.

The changelog entries in `src/App.vue` are the deliberate exception. Each entry is
a historical label for a release that already shipped, so it stays static. Deriving
the newest entry from `package.json` would silently relabel the v2.1.0 entry the
moment the version bumped to v2.2.0.

## Version format

`MAJOR.MINOR.PATCH`, tags prefixed with `v`. Nothing else - no `-alpha` or `-beta`
suffix. The version number plus the git history is the whole story.

| | |
|---|---|
| MAJOR | breaking change to the data model, API, or pipeline contract |
| MINOR | new user-facing capability, backwards compatible |
| PATCH | bug fixes and internal work only |

Note the earlier scheme (`v2.1-beta`) had no patch component and is not valid
SemVer, which left bug-fix-only work with nowhere to land.

## Cutting a release

Bump `package.json` and create the tag **in the same commit**, so the version in the
build and the version in the history can never disagree. This formalizes what the
repo already did ad hoc at `c0f5500` and `092ad00`.

```sh
# 1. bump package.json "version", add the changelog entry in src/App.vue
git commit -am "chore(release): v2.2.0"

# 2. tag that exact commit
git tag -a v2.2.0 -m "v2.2.0"

# 3. publish both
git push origin main --follow-tags
```

Useful afterwards:

```sh
git describe --tags          # which release is HEAD built on top of
git log v2.1.0..HEAD         # what is unreleased right now
git show v2.1.0              # what a given release was
```

## Historical tags

Tags for v1.0.0 through v2.1.0 were backfilled from the in-app changelog
after the fact. They were placed on the last commit at or before each entry's date;
where two entries shared a date, commit content broke the tie. Each tag's annotation
records how it was anchored, so an inferred tag is never mistaken for a certain one:

- **exact** - an explicit version-bump commit exists (v1.8.0, v2.0.0), or
  the commit was the published tip at the time (v2.1.0)
- **content** - the commit subject matches the changelog bullets (v1.3.0,
  v1.4.0, v1.5.0)
- **date** - nearest commit on or before the release date, nothing more specific
  (v1.0.0, v1.1.0, v1.6.0, v1.7.0, v1.9.0)

The early history is coarser than the changelog, so several entries describe work
with no distinct commit behind it. v1.6.0 is a partial case: its commit covers
the rally-segmentation bullet, but nothing in history corresponds to its headline
TrackNetV3 bullet, so it is anchored by date rather than content.

**v1.2.0 has no tag at all** - the extreme case of the same problem. Its entry
is dated February 7, 2026 and describes skeleton/minicourt smoothing work, but the
repository contains no commits whatsoever between February 1 and February 25, 2026.
There is no commit that release could point at. Tagging the nearest earlier commit
would have claimed that work shipped in a commit written before it existed, so the
version was left untagged rather than anchored to something false.
