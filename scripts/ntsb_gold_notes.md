# NTSB CEN10FA443 gold chronology: provenance

`eval/gold/ntsb.json` holds the gold events for the real-document case `ntsb`. The source is the public NTSB docket for CEN10FA443, a Hawker Beechcraft 390 (N6JR) accident at Oshkosh, WI, on July 27, 2010. See `~/case-data/ntsb_cen10fa443/SOURCES.md`.

## How it was built

1. **Reference, not evidence.** The chronology comes from the NTSB final report, `reference/CEN10FA443_final.pdf`, which covers the History of Flight, the pilot and controller interviews and the wreckage sections. The final report is the investigators' conclusion, so it stays out of `evidence/` and the pipeline never sees it.
2. **One supporting evidence file per event.** Each event gives the docket PDF (`doc`) that best supports it. That file is usually the primary record: the ATC factual report for controller transmissions, the CVR report for cockpit audio, the interview transcript for the pilot's own account, and the sheriff report for post-crash actions.
3. **Quotes checked mechanically.** Each `quote` field was located (whitespace-normalized) in the text that `pipeline.parse.parse()` produces for that PDF, with `CT_DISABLE_DOCLING=1`. The script that did this is a scratch helper that is not in the repo. All 27 quotes were found. The `quote`, `date_text` and `provenance` fields are documentation only, because `eval/run_eval.py` scores `date`, `actors` and `description`. `doc` is used for the per-document breakdown.
4. **Dates.** `date` is the calendar day (ISO). Time of day appears in `date_text`. All accident-sequence events fall on 2010-07-27. Investigation events fall on their own days: controller interviews on 07-28, witness statements on 08-02, 08-17 and 08-29, and the pilot interview on 08-30.
5. **Wording.** Descriptions are neutral one-sentence paraphrases written the way an extractor would phrase them, because the eval uses fuzzy description matching (token_set_ratio ≥ 60).

## Coverage

There are 27 events in total.

| Source | Events |
|---|---|
| Final report History of Flight (radio and CVR sequence 1729–1817) | 15 |
| Final report pilot-interview paragraph (go-around decision, throttle position, stall warning) | 4 |
| Post-accident actions (hospital transport, blood draw, site survey) | 3 |
| Investigative events (ATC group meeting, witness statements, pilot interview) | 5 |

Some events have **evidence-only** provenance and do not appear in the final report narrative:
- GPWS "one thousand" at 1815:32
- Lake amphibian sent around at 1817:02
- Sheriff blood draw
- Site measurements on 07-28
- The three witness statements

Each one is marked in `provenance`.

## Evidence notes

- `ATC Group Chairman - FAA Controller Statements-Redacted.PDF` is 11 scanned pages and has no text layer. Its only text is a "(R)" redaction stamp. No gold event depends on it. Until an OCR engine is available, the parser raises `NeedsOCRError`, and the scan reports it in the job error.
- `Statement of Party Representatives to NTSB Investigation-Redacted.PDF` is a boilerplate certification form with no factual events. Any events extracted from it are false positives.
- `Witness Statements-Redacted.PDF` (pages 2–3) contains the same Scheick letter as `ATC Group Chairman - Witness Statement (Stephen J. Scheick, Jr.)-Redacted.PDF`. It is a real cross-document duplicate that merge should link.
- The pilot interview transcript is OCR'd typewriter text, with court-reporter line numbers and the footer "Free State Reporting, Inc." Parsing strips the line numbers and running footers before computing offsets.

## Real contradictions present in the evidence (candidates for a future `ntsb_contradictions.json`)

- **Impact time.** The CVR report records the impact sound at 1816:57. The sheriff summary gives "07/27/10 at 1815 hours", and the Dallenogare supplement gives "07/27/10 at 1817 hours". Witness Ruffin says "about 6:15 p.m.".
- **Cub position.** The controllers and the Soucy witness notes say the Cub was airborne, offset left and "no factor". Scheick believed the yellow airplane "was flying above runway 18R". The pilot recalled the Cub over the runway centerline, as described in the transcript and summarized in the final report.
- **Aircraft acquisition.** In the transcript, the pilot says the airplane was bought in December 2006 with delivery in February 2007. The final report (reference only) gives a purchase date of June 22, 2007. This conflict does not show up when looking at the evidence alone.
