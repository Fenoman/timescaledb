# Early exit for disabled CAGG rewrite traversal — measurement record

**Date:** 2026-08-11
**Branch:** `local_2.29.1`
**Related commits:** `1d0070c64` (original change), `284d187b2` (revert), the
commit introducing this document (restore).

## History and the previously unrecorded revert reason

`1d0070c64` added an early return to `continuous_agg_apply_rewrites()`
(`tsl/src/continuous_aggs/rewrite_with_caggs.c`) when both
`timescaledb.enable_cagg_rewrites` and `timescaledb.cagg_rewrites_debug_info`
are off, plus a `cagg_rewrites` test case pinning the disabled-path plans.

`284d187b2` reverted it the same day because the acceptance benchmark —
planning time of a flat 200×200 `generate_series` join — showed a **6.04%**
median gain against a **≥10%** acceptance gate. That reasoning was recorded
neither in the revert commit message nor anywhere else at the time; this
document records it retroactively, together with the measurements that
superseded the decision.

## Why the change is semantically a no-op

With both GUCs off (their defaults, `src/guc.c:166-167`), a review of
`rewrite_query_with_caggs()` (`rewrite_with_caggs.c:744-978`) shows the walker:

- pins and releases the hypertable cache for every `Query` node it visits
  (`:806` / `:902`) and recurses over the entire tree via `query_tree_walker`
  with `QTW_EXAMINE_RTES_BEFORE | QTW_EXAMINE_RTES_AFTER` and
  `expression_tree_walker`;
- gates every behavioural branch — `check_query_for_cagg_rewrites`,
  `match_query_to_cagg`, all `elog(INFO)` debug output, and setting
  `rewritten_with_caggs` — on one of the two GUCs.

`rewritten_with_caggs` can only be set under `ts_guc_enable_cagg_rewrites`, so
the `QueryRewrite` branch in `continuous_agg_apply_rewrites()` is unreachable
and the function returns `parse` unchanged. The early exit therefore removes
pure overhead. The test case restored together with the code change (EXPLAIN
with both GUCs off, expected outputs for PG 16–19) pins this equivalence.

## Why the original benchmark under-measured

The traversal runs inside the TimescaleDB planner hook
(`src/planner/planner.c:701`) for every planned query while the extension is
loaded, regardless of the GUCs. Its cost is proportional to the number of
`Query` nodes/RTEs in the post-rewrite tree. A flat two-relation join has a
handful of nodes, so the fixed traversal cost stays within noise there. The
Omnix BI workload is the opposite profile: deeply nested view/CTE/subquery
stacks, where the per-node cost accumulates linearly.

## Methodology (2026-08-11)

- Apple Silicon (arm64), macOS 26.5.2; PostgreSQL 16.14 (Homebrew);
  build `build_rebase_pg16_debug` (`CMAKE_BUILD_TYPE=Debug`). Relative deltas
  are the signal; absolute numbers are Debug-inflated.
- Dedicated cluster on port 55434: `shared_preload_libraries=timescaledb`,
  `timescaledb.telemetry_level=off`, `autovacuum=off`, locale `ru_RU.UTF-8`.
- Schema: one hypertable (1000 rows, 1 chunk), one plain `dim` table, and 50
  chained views `v1..v50`, each adding one correlated scalar sublink over
  `dim` — ~101 nested `Query` nodes after view expansion (within the agreed
  50–100 nested view/subquery target).
- Measured statement, both GUCs off (verified in-session):
  `DISCARD PLANS;` then
  `EXPLAIN (SUMMARY ON) SELECT count(*) FROM v50 WHERE device = 3;`
- 30 samples per batch in one warm session; first 5 discarded as warmup,
  25 kept. `Planning Time` parsed from EXPLAIN output.
- A/B/A/B protocol against binary drift: before1 (HEAD) → after1 (patch
  applied) → before2 (patch reversed) → after2 (patch re-applied), with
  rebuild + install + cluster restart between states. Patch equality across
  re-applications verified via `git patch-id --stable`
  (`b68728cdf50e358d7c043186decf6d8fbccc2cf3`, identical to `1d0070c64`).

## Results (Planning Time, ms, n=25 per batch)

| Batch   | State         | Median | Mean  | Stdev | Min   | Max   |
|---------|---------------|--------|-------|-------|-------|-------|
| before1 | traversal on  | 1.331  | 1.317 | 0.101 | 1.081 | 1.492 |
| after1  | early exit    | 1.123  | 1.130 | 0.072 | 1.055 | 1.353 |
| before2 | traversal on  | 1.440  | 1.454 | 0.157 | 1.232 | 1.808 |
| after2  | early exit    | 1.250  | 1.242 | 0.104 | 1.098 | 1.528 |

- Pair 1 (before1 → after1): −0.208 ms, **15.6%** median improvement.
- Pair 2 (before2 → after2): −0.190 ms, **13.2%** median improvement.
- Pooled medians: 1.382 → 1.163 ms (**15.8%**).
- Absolute delta is stable across pairs (≈0.2 ms), i.e. ≈2 µs per `Query`
  node on this machine — consistent with one cache pin/release plus walker
  recursion per node. The saving scales with query-tree size, not with data
  volume.

## Semantics gate

`TESTS=cagg_rewrites cmake --build build_rebase_pg16_debug --target
regresscheck-t` (PG 16, `LC_ALL=ru_RU.UTF-8`): **1/1 passed**, including the
restored disabled-GUC EXPLAIN case.

## Decision

Restore `1d0070c64`. The ≥10% gate that motivated the revert was evaluated on
an unrepresentative flat query; on the nested-view profile the median gain is
13–16%, reproducible across independent rebuild/restart cycles, at zero
semantic risk (provable no-op plus pinned plans). Future perf gates for
planner-hook changes should be measured on the nested-query profile they
actually affect.
