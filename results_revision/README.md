# FROZEN — the artifacts behind the conference revision's new numbers

Copied on 2026-09-29 from `results/revision/` (gitignored) after the runs of the
Day 1–3 work block. They back every number in
`docs/reference/provenance.md` §12 and §12b and in
`paper/conference/revision/text-drafts.md`. **Do not modify.** To regenerate, run
the commands in provenance §12; results go to `results/`, never here.

| File | Produced by | Backs |
|---|---|---|
| `frame_loss_commissioning.json` | `eval.protocol_matrix --clients all --window 400 --stride_exc 5 --bootstrap 2000` | request A, single-loss numbers (first 400 h) |
| `frame_loss_life_gated.json`, `frame_loss_life_sync_safe.json` | `eval.protocol_matrix --clients all --window 0 --boundary_stride 14 --stride_exc 100 --bootstrap 2000` | request A, whole-life confirmation (right-censored) |
| `per_client_v2.json` | `eval.run_eval` (reported config) `--per_client_out` | the per-client vectors behind `stats.json` |
| `stats.json` | `eval.revision_stats` | request B |
| `factorial.json` | `eval.factorial --periods 2,3,4,6 --bootstrap 10000` | request F, and the periodic-assembly correction |
| `tap_split.json` | `eval.tap_split` | request D |
| `batch_daily_cost.json` | inline script, described in the file | §6 row 5 (batch-daily 7.7 %) |
| `training_config.json` | read from `code/dlinear.pt` | request C |
| `sync_safe_cost.json` | `eval.sync_safe_cost --data_dir ../tier2_data --score` (added 2026-10-30, the author's permission) | §IV-E: the sync-safe mode at the operating point's airtime, 0.0136 vs 0.0079; provenance §12c |
| `factorial_lead1.json` | `eval.factorial_lead1` (reads `factorial.json` here; added 2026-10-30) | §VII-D: next-hour error 121.6 vs 133.3 kWh, difference 11.8 (4.9–22.7); provenance §12c |

Not frozen, deliberately: `results/revision/frame_loss_whole_life_gated.json`
(an earlier whole-life run without right-censoring, superseded) and
`results/revision/tier1_results.json` (the per-client run's reported artifact —
identical in every number to the reproduction; `results_frozen/` is the
baseline).

Verify integrity:

    md5sum -c CHECKSUMS.md5
