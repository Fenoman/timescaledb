# Non-FULL database-wide VACUUM bypass — measurement record

**Date:** 2026-08-11
**Branch:** `local_2.29.1`
**Spec:** `docs/superpowers/specs/2026-08-11-vacuum-dbwide-bypass-design.md`
**Related history:** audit Task 8 rejected an ACL-preflight variant (doubled
permission warnings) and an empty-catalog check (never fires in the target
deployment, where every database has at least one hypertable). The approved
approach instead stops intercepting non-FULL database-wide statements
entirely.

## Method

- Apple Silicon (arm64), macOS 26.5.2; PostgreSQL 16.14 (Homebrew); build
  `build_rebase_pg16_debug` (`CMAKE_BUILD_TYPE=Debug`). Relative deltas are
  the signal; absolute numbers are Debug-inflated.
- Dedicated cluster, port 55436: `shared_preload_libraries=timescaledb`,
  `timescaledb.telemetry_level=off`, `autovacuum=off`.
- Database: 5000 empty ordinary tables (`bench_t1..bench_t5000`) plus one
  hypertable (`bench_ht`, 1-day chunks, 240 hourly rows → 10 chunks);
  ~5010 `pg_class` heap relations plus their indexes.
- Measured statement: database-wide `ANALYZE;`, 30 timed executions per
  batch in one warm psql session (client `LC_ALL=C` so `Time:` lines parse
  with dot decimals), first 5 discarded, statistics over the remaining 25.
- A/B/A/B: before1 (bypass reversed) → after1 (applied) → before2 →
  after2, with `git apply`/`-R` of the saved two-hunk patch, incremental
  rebuild, `make install` and cluster restart between states; tree state
  verified via `git diff --quiet` before every batch.

## Results (Time per ANALYZE, ms, n=25 per batch)

| Batch   | State       | Median  | Mean    | Stdev  | Min     | Max     |
|---------|-------------|---------|---------|--------|---------|---------|
| before1 | intercepted | 245.759 | 257.765 | 23.276 | 239.239 | 324.927 |
| after1  | bypass      | 244.192 | 244.810 | 3.482  | 240.258 | 258.054 |
| before2 | intercepted | 244.944 | 246.159 | 4.029  | 240.489 | 257.271 |
| after2  | bypass      | 243.708 | 246.454 | 12.402 | 239.768 | 305.000 |

- Pair 1: −1.567 ms median; pair 2: −1.236 ms median. Direction and
  magnitude agree across independent rebuild/restart cycles.
- Removed work ≈ 0.25–0.32 µs per `pg_class` relation: the full-catalog
  scan with per-tuple ACL checks, hypertable-cache probes and two deep
  copies of the ~5000-entry relation list.
- Honest sizing: ~0.5–0.6% of the end-to-end statement, whose cost is
  dominated by the per-table analyze work itself. The change is a pure
  removal of fixed overhead at zero semantic risk, not a headline speedup.

## Semantics gate

- `TESTS='vacuum vacuum_dbwide vacuum_parallel'` regresscheck (PG16):
  all green in both states; the new solo `vacuum_dbwide` guard output is
  byte-identical between intercepted and bypassed states.
- Bonus behavioral cleanup, found by the guard while it ran as an
  unprivileged role: the intercepted path emits `permission denied to
  analyze ... skipping it` for indexes and TOAST relations because
  upstream's `ts_get_all_vacuum_rels()` checks ACL before filtering
  relkind (vanilla PostgreSQL filters relkind first). A non-superuser
  database-wide `ANALYZE` produced ~270 spurious warnings under
  interception and the vanilla set (2 catalog warnings) under the bypass.
  The guard test now runs as superuser so it pins statistics semantics,
  not the warning set. The check-order bug remains reachable via
  database-wide `VACUUM FULL` and is a candidate for a separate upstream
  fix.

## Decision

Admitted: both pairs improve reproducibly with a stable absolute delta,
and the semantic guard is byte-identical. Landed as one source commit
(code + solo guard test); this record is the accompanying docs commit.
