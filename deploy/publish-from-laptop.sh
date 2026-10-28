#!/usr/bin/env bash
# Build the dashboard on this machine, with every trained model it has, and publish it to OVH.
#
# OVH no longer rebuilds on a timer: it serves the release this script uploads until the next
# one. To have the server recompute the dashboard itself, run on OVH as antonkulaga:
#   sudo systemctl start sugar-data-processing-update.service
set -euo pipefail

dry_run=false
case "${1:-}" in
    "") ;;
    --dry-run) dry_run=true ;;
    *) printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2 ;;
esac

here="$(cd -- "$(dirname -- "$0")" && pwd)"
# Study data, model bundles and the sibling repositories live beside the main checkout,
# also when this script runs from a worktree.
main_repo="$(git -C "$here" worktree list --porcelain | sed -n '1s/^worktree //p')"
host="${SDP_HOST:-antonkulaga@ovh}"
deploy_dir="${SDP_DEPLOY_DIR:-/home/web/ops/sugar-data-processing}"
ref="${SDP_REF:-origin/main}"
input_csv="${SDP_INPUT_CSV:-$main_repo/data/raw/prediction_statistics.csv}"
required_models="${SDP_REQUIRED_MODELS:-sugar_one,citras_w38_2,kronlstm,timestral}"
sugar_root="${SDP_SUGAR_SUGAR:-$main_repo/../sugar-sugar}"
forecasting_root="${GLUCOSE_FORECASTING_DIR:-$main_repo/../glucose-forecasting}"
preview_dir="${SDP_PREVIEW_DIR:-$main_repo/output/real_preview}"
node_bin="${SDP_NODE:-node}"
export SDP_SCORE_CACHE="${SDP_SCORE_CACHE:-$main_repo/data/processed/model_cache/forecasts.parquet}"

for dir in "$sugar_root" "$forecasting_root" "$main_repo/data/raw/hf_models"; do
    if [[ ! -d "$dir" ]]; then
        printf 'Missing %s; the build needs it.\n' "$dir" >&2
        exit 1
    fi
done
test -s "$input_csv"

tmp="$(mktemp -d)"
cleanup() {
    local status=$?
    git -C "$main_repo" worktree remove --force "$tmp/sugar-data-processing" 2>/dev/null || true
    rm -rf -- "$tmp"
    if (( status != 0 )); then
        printf 'Laptop publish failed (exit %s); the dashboard on OVH is unchanged.\n' "$status" >&2
    fi
}
trap cleanup EXIT

git -C "$main_repo" fetch --quiet origin
revision="$(git -C "$main_repo" rev-parse "$ref^{commit}")"
printf 'Building dashboard for %s on this machine.\n' "${revision:0:12}"

# The analysis looks for sugar-sugar and glucose-forecasting beside its own checkout, so the
# clean build checkout gets them as neighbours.
git -C "$main_repo" worktree add --quiet --detach "$tmp/sugar-data-processing" "$revision"
ln -s -- "$(realpath -- "$sugar_root")" "$tmp/sugar-sugar"
ln -s -- "$(realpath -- "$forecasting_root")" "$tmp/glucose-forecasting"
ln -s -- "$main_repo/data/raw/hf_models" "$tmp/sugar-data-processing/data/raw/hf_models"
(
    cd "$tmp/sugar-data-processing"
    uv sync --quiet --locked --no-dev
    uv run --frozen --no-dev sdp analyze --csv "$input_csv" --output "$tmp/output"
)
reports="$tmp/output/reports"
test -s "$reports/explorer.html"
test -s "$reports/milestone.html"
test -s "$reports/analysis_report.json"
python3 - "$reports" "$revision" "$required_models" "$node_bin" "$input_csv" "$sugar_root" <<'PY'
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import subprocess
import sys


class InlineScripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.active = False
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == 'script':
            props = dict(attrs)
            self.active = not props.get('src') and props.get('type', '') in ('', 'text/javascript', 'application/javascript')
            if self.active:
                self.scripts.append('')

    def handle_endtag(self, tag: str) -> None:
        if tag == 'script':
            self.active = False

    def handle_data(self, data: str) -> None:
        if self.active:
            self.scripts[-1] += data


reports = Path(sys.argv[1])
for name in ('explorer.html', 'milestone.html'):
    page = InlineScripts()
    page.feed((reports / name).read_text())
    if not page.scripts:
        raise SystemExit('No dashboard JavaScript found in ' + name)
    for script in page.scripts:
        subprocess.run([sys.argv[4], '--check'], input=script, text=True, check=True)
print('Dashboard JavaScript syntax validated')
report = json.loads((reports / 'analysis_report.json').read_text())
models = sorted({item['model_name'] for item in report['ai']['h6']['comparisons']})
required = {name.strip() for name in sys.argv[3].split(',') if name.strip()}
missing = required.difference(models)
if missing:
    raise SystemExit('Refusing to publish without required model scores: ' + ', '.join(sorted(missing)))
# The study export, its archives and the players' own-data uploads: what the build read.
source = Path(sys.argv[5]).resolve(strict=True)
paths = [source, *sorted(source.parent.glob('archived_*.csv'))]
paths.extend(sorted((Path(sys.argv[6]) / 'data/input/users').glob('*')))
digest = hashlib.sha256()
for path in paths:
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
manifest = {
    'revision': sys.argv[2],
    'input_fingerprint': digest.hexdigest(),
    'published_at': datetime.now(timezone.utc).isoformat(),
    'built_on': 'laptop',
    'models': models,
    'n_runs': report['n_runs'],
    'n_participants': report['n_participants'],
}
(reports / 'deployment.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Validated model scores:', ', '.join(models))
PY

if [[ "$dry_run" == true ]]; then
    rm -rf -- "$preview_dir"
    mkdir -p -- "$preview_dir"
    cp -r -- "$reports/." "$preview_dir/"
    printf 'Dry run: not published. The pages are in %s.\n' "$preview_dir"
    exit 0
fi

release="$(date -u +%Y%m%dT%H%M%SZ)-${revision:0:12}-laptop"
tar -C "$reports" -czf - . | ssh "$host" \
    "sudo -n -u web sh -c 'umask 022 && mkdir -p $deploy_dir/releases/$release.partial && tar -C $deploy_dir/releases/$release.partial -xzf - && mv -T $deploy_dir/releases/$release.partial $deploy_dir/releases/$release'"
ssh "$host" "sudo -n -u web bash -s -- $deploy_dir $release $revision" <<'SH'
set -euo pipefail
deploy_dir="$1"
release="$deploy_dir/releases/$2"
revision="$3"
exec 9>"$deploy_dir/update.lock"
if ! flock -n 9; then
    rm -rf -- "$release"
    printf 'A server update is running; publish again once it ends.\n' >&2
    exit 1
fi
next_link="$deploy_dir/.current.$$"
ln -s -- "$release" "$next_link"
mv -Tf -- "$next_link" "$deploy_dir/current"
printf '%s\n' "$revision" > "$deploy_dir/deployed-revision"
# No server input fingerprint matches a laptop build, so a manual server run rebuilds.
printf 'laptop\n' > "$deploy_dir/deployed-input-fingerprint"
# Keep the current release and two older builds for manual rollback, as the server does.
mapfile -t expired < <(find "$deploy_dir/releases" -mindepth 1 -maxdepth 1 -type d -name '20*' -printf '%f\n' | sort -r | tail -n +4)
for name in "${expired[@]}"; do
    rm -rf -- "${deploy_dir:?}/releases/$name"
done
printf 'Published dashboard at revision %s, built on the laptop.\n' "${revision:0:12}"
SH
