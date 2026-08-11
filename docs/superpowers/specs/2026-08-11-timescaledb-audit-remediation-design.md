# Remediation after the 2.29.1 rebase

**Date:** 2026-08-11
**Base:** `local_2.29.1` at `38f6f7e62`
**Working branch:** `codex/timescaledb-audit-remediation`

## Purpose

Address the confirmed correctness issues and the measured planning/utility
overhead found in the post-rebase audit. Each behavioural change must be an
independently revertible commit. The implementation must not claim support for
an external extension combination that has not passed its integration gate.

## Scope

### Wave 1 — correctness and deterministic regressions

1. Give the Timescale transparent-decompression CustomScan method a unique
   name, `TimescaleDBColumnarScan`, instead of the globally ambiguous
   `ColumnarScan`. Update all internal references and expected plans.
   The integration gate preloads the supplied Citus Columnar fork before
   TimescaleDB and verifies that a parallel compressed-hypertable query has
   the same result as its serial baseline.
2. Port the narrowly scoped upstream fix for incorrect results from min/max
   pushdown under `IS NULL` (`d4283fda851cf39eb134a120b8024e75a7e0ca0b`).
3. Make `ModifyHypertable` runtime EXPLAIN idempotent before `ExecutorEnd`:
   target lists must be hidden and restored exactly once, and emitted counters
   must describe the current executor state rather than grow with the number
   of EXPLAIN polls.
4. Correct SkipScan's per-group offset estimate from `1 / selectivity` to
   `1 / selectivity - 1` with a non-negative lower bound.

### Wave 2 — benchmark-gated overhead reductions

Each item begins with a reproducible baseline. It is committed only if its
functional tests pass and the benchmark shows no regression for ordinary
relations or supported hypertable queries.

5. Bypass continuous-aggregate rewrite work when both its functional and
   debug controls are disabled.
6. Avoid Timescale's full catalog enumeration for database-wide VACUUM when
   no relevant hypertable or continuous aggregate exists; otherwise preserve
   the current interception path.
7. Remove duplicate per-query relation/chunk classification probes by sharing
   planner-local classification data without persisting it across a query.
8. Replace per-chunk compression-settings catalog lookups with a
   planner-local bulk lookup only if the benchmark confirms an improvement and
   preserves per-chunk overrides.
9. Correct or remove the per-child quadratic ColumnarScan sort-planning cost
   only after a plan/cost benchmark proves that the replacement avoids the
   current cubic aggregate penalty without causing harmful plan flips.

## Invariants

- A Timescale CustomScan must serialize and deserialize to Timescale methods
  regardless of `shared_preload_libraries` order.
- Query results, executor lifecycle, SQL API, on-disk catalog format, and
  extension upgrade paths remain unchanged except for the corrected bugs.
- A disabled CAGG rewrite is observationally inert.
- A VACUUM that does not involve Timescale objects follows PostgreSQL's normal
  utility path without duplicate relation enumeration.
- Planner caches are bounded by the planning invocation and cannot become
  stale across statements or transactions.
- Ordinary SELECT, DML, and DDL without hyper tables remain eligible for the
  PostgreSQL plan and utility paths they used before the change.

## TDD and commit model

For every source change:

1. Add or extend a regression/integration test that fails against the parent
   commit for the stated defect or benchmark guard.
2. Run it and record the red result.
3. Implement the smallest scoped change.
4. Run the covering test green, the relevant build, `git diff --check`, and a
   focused review.
5. Commit exactly that fix and its tests. Do not fold unrelated expected-file
   cleanup into a behavioural commit.

The design document and implementation plan are documentation commits; every
actual source fix remains a separate commit.

## Validation matrix

- Targeted regression coverage for every fix on PostgreSQL 16, 17, and 18.
- A fresh PG16 Citus Columnar build from
  `/Users/fenoman/GitProjects/pgpro_citus` is used only in an external,
  non-installing integration gate. It must not modify that checkout.
- Wave 2 uses ordinary-table, 10-chunk, 100-chunk, and 1000-chunk workloads
  where applicable. Record planning time, chosen plan, and execution result.
- The final matrix runs every locally available supported suite. Missing TAP
  dependencies in Homebrew bottle installations are explicitly reported as
  unavailable rather than treated as passing.

## Explicit non-goals and external gates

- `pgpro_stats` declares this hook-heavy extension stack unsupported. This
  series removes the confirmed CustomScan collision but does not assert that
  it makes the complete stack vendor-supported.
- `pg_xclaim` transaction-callback ordering, exact Postgres Pro ABI branches,
  and broad `first`/`last` aggregate coalescing need separate integration or
  design work. They are not silently changed in this series.
- If a Wave 2 proposal lacks a reproducible improvement or a safe TDD route,
  it is documented as deferred rather than merged speculatively.

## Completion criteria

The branch contains one reviewed commit per accepted fix, all targeted tests
are green, the final multi-version matrix and Citus integration gate have
recorded results, and any unavailable external validation is listed with its
exact blocker.
