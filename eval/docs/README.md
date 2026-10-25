# Public evaluation documents

`manifest.json` lists the public documents for the SPEC §13 eval set: the Twitter v. Musk
letters (Musk's Schedule 13D/A exhibits and Twitter's 8-K exhibits on EDGAR) and pleadings,
the SEC and CFTC FTX complaints, two Rule 56.1 statement-and-response pairs (SEC v. Ripple Labs;
Freeman v. Giuliani), one Enron email thread (DOJ trial exhibit), and one deposition excerpt.
Every URL was fetched and checked by hand; the `verified` field records when and what was seen.
Things we could not find yet are under `todo`.

```bash
make fetch-eval                                  # needs internet: run in setup mode
uv run python scripts/fetch_eval.py --only ftx   # one group
```

Files are saved to `eval/docs/<group>/` and are git-ignored. The first fetch writes each file's
`sha256` and `retrieved_at` into the manifest; commit that change so later fetches are verified.
EDGAR `.htm` exhibits also get a `.txt` rendition, which is what the pipeline reads. Two
pleadings (`official: false`) come from news-site copies because the Delaware Court of Chancery
does not publish filings.

Each group is meant to be scanned as its own case:

```bash
CT_EVIDENCE_DIR=eval/docs/ftx CT_CASE_ID=ftx make scan CASE=ftx
make eval CASE=ftx
```

## Gold labels: to be made by hand

`eval/gold_timeline.json` and `eval/gold_disputes.json` are intentionally empty (`[]`). They must
be labelled by people reading these documents; do not generate them with the model, or the eval
measures the model against itself.

- `gold_timeline.json`: one entry per fact per document, `{"id", "title", "details"?, "date",
  "date_end"?, "date_certainty", "doc", "event_key", "quote", "people"}`. `doc` is the filename
  the pipeline sees (the `.txt` rendition for EDGAR exhibits); `quote` is verbatim; entries about
  the same event in different documents share an `event_key` (this drives merge accuracy).
- `gold_disputes.json`: from the Rule 56.1 pairs, `{"id", "label": "disputed" | "undisputed",
  "field"?, "sides": [{"doc", "quote"}, ...]}`, one entry per numbered paragraph, with the
  movant's statement and the response as the two sides.

The fictional demo case keeps its gold under `eval/gold/demo_timeline.json` and
`eval/gold/demo_disputes.json`. `eval/run_eval.py` looks for `eval/gold/<case>_timeline.json`
first and falls back to the public-set files.
