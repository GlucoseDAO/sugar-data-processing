# OVH dashboard updates

The dashboard at `https://data.sugar-sugar.study/` is served directly by Caddy.
The server checks GitHub `origin/main` every five minutes using the systemd
timer below. No Codex session or running laptop is needed.

The script runs as `web`, fast-forwards a clean checkout, installs the locked
production dependencies, and builds from the configured study CSV. With no
configuration it fetches the sibling production export. OVH uses a prepared,
linked export so refreshing the code cannot replace person-level identities
with browser IDs. If neither GitHub nor the input/model files have changed,
it exits without rebuilding. Failed builds are
retried on the next timer run, even if Git has already advanced. Local Git
changes or diverged history stop the update without overwriting files.

Reports are built in a temporary directory. Once the build succeeds, an atomic
`current` symlink change publishes it. Three releases are retained; Caddy needs
no reload for subsequent updates. Raw data and intermediate tables are outside
the public reports directory.

## Installation on OVH

The repository lives at `/home/web/sugar-data-processing`. Install the script
outside the checkout so it can update the repository without changing the
script that is currently running:

```bash
sudo install -d -o web -g web /home/web/ops/sugar-data-processing
sudo install -o web -g web -m 755 deploy/update-dashboard.sh \
  /home/web/ops/sugar-data-processing/update-dashboard.sh
sudo install -m 644 deploy/sugar-data-processing-update.service \
  deploy/sugar-data-processing-update.timer /etc/systemd/system/
sudo -u web -H /home/web/ops/sugar-data-processing/update-dashboard.sh --force
```

After the first successful build, the Caddy site block should use:

```caddyfile
data.sugar-sugar.study {
    encode zstd gzip
    root * /home/web/ops/sugar-data-processing/current
    header Cache-Control "no-cache"
    file_server {
        index explorer.html
    }
}
```

Back up the existing Caddyfile, validate it as `caddy`, and reload Caddy. Then
enable the timer:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sugar-data-processing-update.timer
```

## Operations

Check immediately:

```bash
sudo systemctl start sugar-data-processing-update.service
```

Rebuild even when GitHub has not changed (for example, to refresh study data):

```bash
sudo -u web -H /home/web/ops/sugar-data-processing/update-dashboard.sh --force
```

Inspect the schedule and logs:

```bash
systemctl list-timers sugar-data-processing-update.timer
journalctl -u sugar-data-processing-update.service -n 50 --no-pager
```

Disable automatic updates:

```bash
sudo systemctl disable --now sugar-data-processing-update.timer
```

## Prepared study data and trained models

OVH reads `/home/web/ops/sugar-data-processing/update.env`:

```dotenv
SDP_INPUT_CSV=/home/web/ops/sugar-data-processing/inputs/prediction_statistics.csv
SDP_REQUIRED_MODELS=citras,citras_w38_2,kronlstm
```

Keep the scrubbed, linked export and its `archived_*.csv` files in the `inputs`
directory. This is a reviewed data snapshot: refresh it explicitly when a new
linked study export is ready. The updater does not overwrite it with a fetch
of the raw browser-ID export. Keep trained ONNX bundles in the repository's
gitignored `data/raw/hf_models/` directory, as on the local analysis machine.
No Hugging Face credential is needed when the bundles are already on disk.

The updater fingerprints the CSVs, model bundles, and installed script. Input
changes trigger a rebuild even at the same Git commit. A build is published
only if the configured trained models appear in its scores; otherwise the
existing dashboard stays live. `/deployment.json` records the published Git
revision, input fingerprint, models, and cohort counts without contact fields.

The defaults can be overridden with `SDP_REPO`, `SDP_DEPLOY_DIR`, `SDP_BRANCH`,
`SDP_UV`, `SDP_INPUT_CSV`, and `SDP_REQUIRED_MODELS`. A change to the installed script or units must be installed again;
pulling application commits does not replace the running deployment tooling.
