# Anonymized statistics export

This folder holds the CSV evaluators use to rebuild the study HTML. The file is produced locally and is not committed.

Store the file you were given at:

`data/anonymized_raw/prediction_statistics.csv`

From a checkout of this repo:

```bash
uv sync
uv run sdp analyze --csv data/anonymized_raw/prediction_statistics.csv
```

Reports are written to `output/reports/`. `milestone.html` is the human page.

Regenerate the CSV from the last server pull:

```bash
uv run sdp anonymize
```

The server pull and the column list are in the repo README, under "What is extracted from the server".
