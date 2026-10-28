#!/usr/bin/env bash
# Check GitHub, build changed revisions, and atomically publish the dashboard.
set -euo pipefail

force=false
case "${1:-}" in
    "") ;;
    --force) force=true ;;
    *) printf 'Usage: %s [--force]\n' "$0" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    printf 'Usage: %s [--force]\n' "$0" >&2
    exit 2
fi

repo="${SDP_REPO:-/home/web/sugar-data-processing}"
deploy_dir="${SDP_DEPLOY_DIR:-/home/web/ops/sugar-data-processing}"
branch="${SDP_BRANCH:-main}"
uv="${SDP_UV:-/home/web/.local/bin/uv}"
build_dir=""
next_link=""

cleanup() {
    local status=$?
    if [[ -n "$build_dir" ]]; then rm -rf -- "$build_dir"; fi
    if [[ -n "$next_link" ]]; then rm -f -- "$next_link"; fi
    if (( status != 0 )); then
        printf 'Dashboard update failed (exit %s); the last published dashboard is still served.\n' "$status" >&2
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p "$deploy_dir/releases"
exec 9>"$deploy_dir/update.lock"
if ! flock -n 9; then
    printf 'Another dashboard update is already running.\n'
    exit 0
fi

cd "$repo"
if [[ "$(git branch --show-current)" != "$branch" ]]; then
    printf 'Expected branch %s; refusing to switch branches automatically.\n' "$branch" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    printf 'Server checkout has local changes; refusing to update it.\n' >&2
    exit 1
fi

export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o ConnectTimeout=15}"
git fetch --quiet origin "refs/heads/$branch:refs/remotes/origin/$branch"
revision="$(git rev-parse "refs/remotes/origin/$branch")"
git merge --ff-only "$revision"
if [[ "$(git rev-parse HEAD)" != "$revision" ]]; then
    printf 'Server branch has commits beyond GitHub; refusing to publish them automatically.\n' >&2
    exit 1
fi

deployed=""
if [[ -f "$deploy_dir/deployed-revision" ]]; then
    deployed="$(cat "$deploy_dir/deployed-revision")"
fi
if [[ "$force" == false && "$revision" == "$deployed" && -s "$deploy_dir/current/explorer.html" ]]; then
    printf 'No GitHub changes; dashboard is already at %s.\n' "${revision:0:12}"
    exit 0
fi

printf 'Building dashboard for %s.\n' "$revision"
"$uv" sync --locked --no-dev
"$uv" run --frozen --no-dev sdp fetch --sibling
build_dir="$(mktemp -d "$deploy_dir/.build.XXXXXXXX")"
"$uv" run --frozen --no-dev sdp analyze --output "$build_dir/output"
test -s "$build_dir/output/reports/explorer.html"
test -s "$build_dir/output/reports/milestone.html"
test -s "$build_dir/output/reports/analysis_report.json"

release="$deploy_dir/releases/$(date -u +%Y%m%dT%H%M%SZ)-${revision:0:12}"
mv -- "$build_dir/output/reports" "$release"
next_link="$deploy_dir/.current.$$"
ln -s -- "$release" "$next_link"
mv -Tf -- "$next_link" "$deploy_dir/current"
next_link=""
printf '%s\n' "$revision" > "$deploy_dir/deployed-revision"
printf 'Published dashboard at revision %s.\n' "${revision:0:12}"

# Keep the current release and two older builds for manual rollback.
mapfile -t expired < <(find "$deploy_dir/releases" -mindepth 1 -maxdepth 1 -type d -name '20*' -printf '%f\n' | sort -r | tail -n +4)
for name in "${expired[@]}"; do
    rm -rf -- "${deploy_dir:?}/releases/$name"
done
