# OVH dashboard updates

The dashboard at `https://data.sugar-sugar.study/` is static pages served
directly by Caddy. Since 28 October 2026 they are built on the analysis laptop,
which has every trained model, and uploaded with `deploy/publish-from-laptop.sh`
(see below). OVH no longer rebuilds on its own: its timer is disabled, and the
server rebuilds only when its update service is started by hand.

The server's update script runs as `web`, fast-forwards a clean checkout, installs the locked
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

## Publishing from the laptop

```bash
deploy/publish-from-laptop.sh
```

The script builds GitHub `origin/main` in a clean temporary checkout, so other
work in the local checkout never reaches the dashboard. It reads, beside the main
checkout:

- `data/raw/prediction_statistics.csv` and its `archived_*.csv`: the same
  prepared, linked export as OVH's `inputs/` directory, not a raw browser-ID fetch;
- `../sugar-sugar/data/input/users`: the players' own-data uploads (gitignored);
- `data/raw/hf_models/`: the CITRAS-FM W38.2, KronLSTM R.80 and Timestral R.81 bundles;
- `../glucose-forecasting` (or `GLUCOSE_FORECASTING_DIR`): SugarOne 1.0.

A build is published only if it scored `sugar_one`, `citras_w38_2`, `kronlstm` and
`timestral` (override with `SDP_REQUIRED_MODELS`) and both pages pass the Node.js
syntax check. It is uploaded as `releases/<time>-<revision>-laptop` and the `current`
symlink is switched, as the server does; three releases are kept. `--dry-run` builds
and checks without uploading and leaves the pages in `output/real_preview/`.
Forecasts are cached in the main checkout's `data/processed/model_cache/`, so a
rebuild scores only new or changed game windows.

Players keep uploading their own data to the game on OVH. Copy the uploads to the
laptop before publishing new study data:

```bash
ssh antonkulaga@ovh 'sudo -n tar -C /home/web/sugar-sugar/data/input/users -czf - .' | tar -C ../sugar-sugar/data/input/users -xzf -
```

## Operations

Rebuild on the server by hand, as `antonkulaga` (`web` has no sudo). This
publishes a server build, which has only the models on the server (CITRAS-FM W38.2
and KronLSTM R.80), in place of the laptop build:

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

Automatic updates are disabled. To turn them back on, or off again:

```bash
sudo systemctl enable --now sugar-data-processing-update.timer
```

```bash
sudo systemctl disable --now sugar-data-processing-update.timer
```

## Prepared study data and trained models

OVH reads `/home/web/ops/sugar-data-processing/update.env`:

```dotenv
SDP_INPUT_CSV=/home/web/ops/sugar-data-processing/inputs/prediction_statistics.csv
SDP_REQUIRED_MODELS=citras_w38_2,kronlstm
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
Both generated HTML pages also pass a Node.js JavaScript syntax check before
publication. Set `SDP_NODE` to the Node executable when it is not on the service's
PATH (OVH uses `/home/web/.nvm/versions/node/v22.23.1/bin/node`).

Model forecasts are cached in the checkout's gitignored
`data/processed/model_cache/forecasts.parquet`, outside the temporary build
directory, so a rebuild scores only new or changed game windows. The first build
after new model bundles scores everything. Add `SDP_SCORE_CACHE=off` to
`update.env` (or delete the file) to force a full rescore.

The defaults can be overridden with `SDP_REPO`, `SDP_DEPLOY_DIR`, `SDP_BRANCH`,
`SDP_UV`, `SDP_INPUT_CSV`, `SDP_REQUIRED_MODELS`, and `SDP_NODE`. A change to the installed script or units must be installed again;
pulling application commits does not replace the running deployment tooling.
