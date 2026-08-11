# Per-chunk compression-settings lookup — measurement record

**Date:** 2026-08-11
**Branch:** `local_2.29.1` at `bd7bd422e`
**Question:** is a bulk (single-scan, per-query hash) replacement for the
per-chunk `ts_compression_settings_get()` point lookup in
`build_compressioninfo()` (`tsl/src/nodes/columnar_scan/columnar_scan.c:679`)
worth building?
**Disposition:** deferred by owner decision (2026-08-11) — see verdict.

## Method

- Dedicated PG16.14 (Homebrew) cluster, port 55435,
  `shared_preload_libraries=timescaledb`, `autovacuum=off`, telemetry off,
  extension at `bd7bd422e` (Debug build; relative shares are the signal).
- Hypertable with 1-hour chunks, 1 row/chunk, 1000 chunks, all compressed
  (`compress_segmentby='device'`); planning time from
  `EXPLAIN (SUMMARY ON) SELECT count(*)` over 10 / 100 / 1000 chunks;
  30 samples per point, first 5 discarded, medians over 25, `DISCARD PLANS`
  between samples.
- Backend call-stack shares via macOS `/usr/bin/sample` (2-3 runs of 10 s
  against a 200-iteration EXPLAIN loop), symbols resolved.
- Catalog-density variant: second compressed hypertable brings
  `_timescaledb_catalog.compression_settings` from ~1001 to ~2002 rows.

## Results

| chunks | planning median, `max_files_per_process=1000` | `=10000` |
|--------|-----------------------------------------------|----------|
| 10     | 0.236 ms                                      | —        |
| 100    | 2.251 ms                                      | 2.515 ms |
| 1000   | 107.069 ms                                    | 22.982 ms |

- Clean linear fit (fd-safe points): **23.0 µs/chunk** total planning cost,
  intercept ≈ 0.
- `ts_compression_settings_get` inclusive share of `pg_plan_query`:
  **10.4–10.8%** with a healthy fd cache (≈2.5 ms at 1000 chunks;
  ≈2.4–3.0 µs per compressed chunk). Cost split ≈ 1/3 catalog
  `table_open`+lock, 1/3 btree `getnext`, 1/4 `endscan`/unlock.
- Catalog density 1001 → 2002 settings rows: no measurable effect (same
  btree depth; lookup dominated by open/lock/scan setup).

## Incidental findings (larger than the question asked)

1. **vfd-cache thrashing dominates at high chunk counts.** With the default
   `max_files_per_process=1000`, 1000 compressed chunks (≈4000 files: chunk,
   compressed chunk, and their indexes) exceed the vfd cache and planner
   size probes (`mdnblocks` → `FileSize` → `open()`) reopen files every
   planning cycle: 107 ms vs 23 ms — **4.7×**. Deployment-side remedy
   (raise `max_files_per_process`, align OS `nofile`) was handed to the
   `postgrespro-docker` project.
2. Relation size probes in `get_relation_info` account for ≈19% of clean
   planning time — roughly twice the settings lookups; a larger future
   target than bulk settings lookup.

## Verdict

Bulk lookup is justified only for workloads that regularly plan
**≥300–500 compressed chunks per query** (saves ~2.5–3 ms / ~10% at 1000
chunks; <0.25 ms at 100). Deferred until such workloads are demonstrated in
Omnix; revisit with this record as the baseline. Raw series and profiles were
captured in the session workspace (`csbench-report.md`, `bench/sample*.txt`)
and summarized here.
