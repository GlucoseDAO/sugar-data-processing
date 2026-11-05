# Anonymized statistics export

This folder is what evaluators use to rebuild the study HTML. The files are produced locally and are not committed.

Put both CSVs in this folder and run them together:

```
data/anonymized_raw/prediction_statistics.csv
data/anonymized_raw/prediction_statistics_ai.csv
```

From a checkout of this repo:

```bash
uv sync
uv run sdp analyze --csv data/anonymized_raw/prediction_statistics.csv
```

`analyze` reads the human file and loads `prediction_statistics_ai.csv` from the same folder. Reports go to `output/reports/`. `explorer.html` is the full page. `milestone.html` is the human page.

If you already have the anonymized human file, drop the AI CSV next to it and re-run the same command. `study_id` is unchanged, so the join works. The HTML still builds without the AI file; you just do not get the in-game forecasts.

Regenerate the pack from the last server pull:

```bash
uv run sdp anonymize
```

The server pull and the column list are in the repo README, under "What is extracted from the server".
