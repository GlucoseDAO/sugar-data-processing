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
input_csv="${SDP_INPUT_CSV:-}"
required_models="${SDP_REQUIRED_MODELS:-}"
script_path="$(readlink -f -- "$0")"
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
source_csv="${input_csv:-$repo/../sugar-sugar/data/input/prediction_statistics.csv}"
input_fingerprint="$(python3 - "$source_csv" "$repo" "$required_models" "$script_path" <<'PY'
import hashlib
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve(strict=True)
repo = Path(sys.argv[2])
paths = [source, *sorted(source.parent.glob('archived_*.csv'))]
models = repo / 'data/raw/hf_models'
paths.extend(sorted(models.glob('*/*.onnx')))
paths.extend(sorted(models.glob('*/*.json')))
paths.append(Path(sys.argv[4]).resolve(strict=True))
digest = hashlib.sha256(sys.argv[3].encode())
for path in paths:
    digest.update(str(path).encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
print(digest.hexdigest())
PY
)"
deployed_inputs=""
if [[ -f "$deploy_dir/deployed-input-fingerprint" ]]; then
    deployed_inputs="$(cat "$deploy_dir/deployed-input-fingerprint")"
fi
if [[ "$force" == false && "$revision" == "$deployed" && "$input_fingerprint" == "$deployed_inputs" && -s "$deploy_dir/current/explorer.html" ]]; then
    printf 'No GitHub or input changes; dashboard is already at %s.\n' "${revision:0:12}"
    exit 0
fi

printf 'Building dashboard for %s.\n' "$revision"
"$uv" sync --locked --no-dev
if [[ -n "$input_csv" ]]; then
    printf 'Using prepared analysis input: %s\n' "$input_csv"
else
    "$uv" run --frozen --no-dev sdp fetch --sibling
    input_csv="$repo/data/raw/prediction_statistics.csv"
fi
build_dir="$(mktemp -d "$deploy_dir/.build.XXXXXXXX")"
"$uv" run --frozen --no-dev sdp analyze --csv "$input_csv" --output "$build_dir/output"
test -s "$build_dir/output/reports/explorer.html"
test -s "$build_dir/output/reports/milestone.html"
test -s "$build_dir/output/reports/analysis_report.json"
python3 - "$build_dir/output/reports" "$revision" "$input_fingerprint" "$required_models" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

reports = Path(sys.argv[1])
report = json.loads((reports / 'analysis_report.json').read_text())
models = sorted({item['model_name'] for item in report['ai']['h6']['comparisons']})
required = {name.strip() for name in sys.argv[4].split(',') if name.strip()}
missing = required.difference(models)
if missing:
    raise SystemExit('Refusing to publish without required model scores: ' + ', '.join(sorted(missing)))
manifest = {
    'revision': sys.argv[2],
    'input_fingerprint': sys.argv[3],
    'published_at': datetime.now(timezone.utc).isoformat(),
    'models': models,
    'n_runs': report['n_runs'],
    'n_participants': report['n_participants'],
}
(reports / 'deployment.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Validated model scores:', ', '.join(models))
PY

release="$deploy_dir/releases/$(date -u +%Y%m%dT%H%M%SZ)-${revision:0:12}"
mv -- "$build_dir/output/reports" "$release"
next_link="$deploy_dir/.current.$$"
ln -s -- "$release" "$next_link"
mv -Tf -- "$next_link" "$deploy_dir/current"
next_link=""
printf '%s\n' "$revision" > "$deploy_dir/deployed-revision"
printf '%s\n' "$input_fingerprint" > "$deploy_dir/deployed-input-fingerprint"
printf 'Published dashboard at revision %s.\n' "${revision:0:12}"

# Keep the current release and two older builds for manual rollback.
mapfile -t expired < <(find "$deploy_dir/releases" -mindepth 1 -maxdepth 1 -type d -name '20*' -printf '%f\n' | sort -r | tail -n +4)
for name in "${expired[@]}"; do
    rm -rf -- "${deploy_dir:?}/releases/$name"
done
