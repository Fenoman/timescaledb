# Bypass Timescale interception for non-FULL database-wide VACUUM

**Date:** 2026-08-11
**Branch:** `local_2.29.1`
**Status:** approved (approach A of A/B/C; owner picked A on 2026-08-11)

## Purpose

Remove the fixed per-statement overhead that TimescaleDB adds to database-wide
`VACUUM` / `ANALYZE` (a `VacuumStmt` with `rels == NIL`). The deployment
profile this targets runs a scheduled database-wide `VACUUM ANALYZE` /
`ANALYZE` daily on databases that always contain at least one hypertable, so
the earlier "skip when no hypertable exists" idea (audit Task 8) never fires
and is not pursued.

## Findings the design rests on

In `process_vacuum()` (`src/process_utility.c`), the `stmt->rels == NIL`
branch:

1. `ts_get_all_vacuum_rels()` performs a full `pg_class` heap scan with a
   per-tuple ACL check, mirroring PostgreSQL's own `get_all_vacuum_rels()`.
   Its only Timescale-specific side effect is
   `register_chunk_for_rebuild_if_needed()`, which returns immediately unless
   the statement is `VACUUM FULL` (`ctx.is_vacuumfull`).
2. `ctx.chunk_rels` stays `NIL` in this branch; chunk expansion, continuous
   aggregate redirection, and compressed-chunk additions happen only in the
   explicit-relation (`rels != NIL`) branch. Chunks and compressed chunks are
   ordinary `pg_class` rows, so the database-wide list already contains them.
3. The resulting `exec_rels` handed to `ExecVacuum()` is therefore exactly
   the list PostgreSQL would build for itself, in the same `pg_class` scan
   order, and `vacuum_rel()` / `analyze_rel()` re-run all permission checks
   anyway.

Hence, for non-FULL database-wide statements the interception produces zero
semantic difference while paying: one full `pg_class` scan with ACL checks,
hypertable-cache probes (including negative-entry creation on first touch),
and two deep copies of the relation list.

## Change

At the top of `process_vacuum()`, after the options loop has computed
`ctx.is_vacuumfull` and handled `only_database_stats`:

```c
if (stmt->rels == NIL && !ctx.is_vacuumfull)
    return DDL_CONTINUE;
```

PostgreSQL then executes the statement through its normal utility path.

## What is preserved unchanged

- `VACUUM FULL` database-wide: interception retained; it is what feeds
  `rebuild_columnstore_chunk_oids` (missing-attribute materialization must
  propagate to columnstore data).
- Explicit-relation statements (`rels != NIL`): continuous-aggregate
  redirection, hypertable chunk expansion, direct-chunk compressed-relation
  handling, rebuild registration.
- `VACUUM (ONLY_DATABASE_STATS)`: already returns `DDL_CONTINUE`.

## Semantics argument

The bypass makes behavior strictly more vanilla, not just equivalent: today a
relation can lose privileges (or be dropped) between Timescale's enumeration
and `ExecVacuum()`'s re-check, producing skip warnings that a single-pass
vanilla execution would not emit. With the bypass, PostgreSQL performs its
single enumeration itself and those windows disappear. No catalog format, SQL
API, or upgrade-path impact.

## Tests (semantic guard, `test/sql/vacuum.sql`)

- Database-wide `VACUUM ANALYZE` with a populated hypertable and a populated
  ordinary table present: both the ordinary table and the chunks show updated
  `pg_stat_all_tables` counters (proves chunks are still processed via the
  vanilla path).
- Database-wide `ANALYZE`: same assertion for statistics targets.
- Existing explicit-relation and `VACUUM FULL` tests must pass unchanged.

Tests are added first and must pass against the parent commit too (the guard
asserts unchanged behavior; the red gate for this change is the benchmark
baseline, per the Wave-2 convention of the audit series).

## Benchmark gate

A/B/A/B protocol as in
`docs/superpowers/benchmarks/2026-08-11-cagg-rewrites-disabled-early-exit.md`:
dedicated PG16 cluster, disposable database with ~5000 empty ordinary tables
plus one hypertable (maximizes the enumeration share), 25 kept samples per
batch of database-wide `ANALYZE` wall time via `\timing`, rebuild + reinstall
+ restart between states, medians recorded in a
`docs/superpowers/benchmarks/` record. Admission: reproducible median
improvement across both pairs and a zero diff on the semantic guard tests.
If the gate fails, record the numbers and land no source commit.

## Risks

- Divergence from upstream in this branch of `process_vacuum()`: the insertion
  point is two lines at a stable location; rebase conflicts are trivial. The
  fork already diverges in this function (ownership isolation, ordering
  hardening).
- None of the code involved is PG-major-specific; PG 16/17/18 behave
  identically.

## Non-goals

- No change to the `VACUUM FULL` enumeration cost (hashset inversion —
  approach B — rejected as YAGNI for this profile).
- Bulk compression-settings lookup: measured separately and deferred; see
  `docs/superpowers/benchmarks/2026-08-11-compression-settings-lookup.md`.

## Completion criteria

One source commit (guard tests + code), its benchmark record committed
separately, targeted `vacuum` regresscheck green on PG16/17/18 local builds,
`git diff --check` clean but for known expected-output artifacts.
