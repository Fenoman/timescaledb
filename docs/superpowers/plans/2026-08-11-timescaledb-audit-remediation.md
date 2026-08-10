# TimescaleDB Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Correct confirmed post-rebase defects and land only planning/utility optimizations whose measured admission gates prove a benefit without changing PostgreSQL semantics.

**Architecture:** The series gives transparent decompression a unique CustomScan identity, ports the upstream NULL/minmax correctness fix, makes ModifyHypertable EXPLAIN repeatable, and corrects a SkipScan cost formula. Planning/utility changes are isolated from correctness work: each has a semantic guard and a quantified pre/post benchmark; a candidate that misses its gate is documented as deferred and receives no source commit.

**Tech Stack:** C17, PostgreSQL CustomScan and hook APIs, CMake, pg_regress, Perl TAP, PostgreSQL 16/17/18, Homebrew PostgreSQL, and the local pgpro Citus Columnar fork.

## Global Constraints

- Work only in /Users/fenoman/GitProjects/timescaledb/.worktrees/timescaledb-audit-remediation on codex/timescaledb-audit-remediation, based on 38f6f7e62.
- Each accepted source fix, its tests, and expected output form exactly one independently revertible commit. Documentation and benchmark records are separate.
- Correctness tasks must add their test first, execute the parent code to show red, then apply the smallest code change.
- Pure overhead work uses a recorded median planning-time baseline as its red gate; code may be committed only after the threshold and semantic guard pass.
- Keep the early ExecutorStart EXPLAIN test from db24fd8cf unchanged. Repeated EXPLAIN coverage uses a distinct ExecutorRun hook.
- The Timescale CustomScan and CustomPath name is exactly TimescaleDBColumnarScan.
- Build Citus outside /Users/fenoman/GitProjects/pgpro_citus. Never make install and never alter or delete that checkout.
- Targeted tests run on PG16/17/18 when configured. The final matrix reports missing Homebrew TAP prerequisites as unavailable, not passing.
- Do not claim that pgpro_stats, pg_xclaim, or a Postgres Pro ABI branch becomes supported from this code series.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| tsl/src/nodes/columnar_scan/planner.c | CustomScan method name and plan construction |
| tsl/src/nodes/columnar_scan/columnar_scan.c | CustomPath method name and sort cost |
| tsl/src/nodes/columnar_scan/qual_pushdown.c | Exact versus lossy qual pushdown |
| src/nodes/modify_hypertable.c and .h | EXPLAIN lifecycle and counter snapshots |
| tsl/src/nodes/skip_scan/planner.c | Compressed SkipScan cost |
| tsl/src/continuous_aggs/rewrite_with_caggs.c | CAGG rewrite entry |
| src/process_utility.c | Database-wide VACUUM interception |
| src/planner/planner.c and .h | Query-lifetime relation/chunk classification |
| test/src/test_utils.c and test/sql/test_utils.sql | Debug executor-hook regression helper |
| tsl/test/t/006_citus_columnar_customscan.pl | Opt-in Citus parallel serialization gate |

### Task 1: Establish the worktree-owned PG16 baseline

**Files:**
- Create: ignored build-pg16-debug directory.
- Modify: none.
- Test: selected core and TSL schedules.

**Interfaces:**
- Consumes: /opt/homebrew/opt/postgresql@16/bin/pg_config and /tmp/timescaledb-testdeps-20260810/pg-src/postgresql-16.14.
- Produces: a Debug build bound to this worktree and a baseline.md record in this plan's SDD workspace.

- [ ] **Step 1: Verify the clean source and matching PostgreSQL source tree**

Run:

~~~bash
git status --short
test -d /tmp/timescaledb-testdeps-20260810/pg-src/postgresql-16.14
/opt/homebrew/opt/postgresql@16/bin/pg_config --version
~~~

Expected: no tracked changes besides committed planning artifacts and PostgreSQL 16.14.

- [ ] **Step 2: Configure and build the Debug target**

Run:

~~~bash
cmake -S . -B build-pg16-debug \
  -DPG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config \
  -DPG_SOURCE_DIR=/tmp/timescaledb-testdeps-20260810/pg-src/postgresql-16.14 \
  -DCMAKE_BUILD_TYPE=Debug -DASSERTIONS=ON \
  -DREGRESS_CHECKS=ON -DREQUIRE_ALL_TESTS=ON
cmake --build build-pg16-debug -j2 --target timescaledb timescaledb-tsl
~~~

Expected: both Timescale libraries build; a missing TAP dependency is a concrete blocker recorded in baseline.md.

- [ ] **Step 3: Run the narrow parent-commit baseline**

Run:

~~~bash
TESTS='test_utils baserel_cache vacuum' \
  cmake --build build-pg16-debug --target regresscheck
TESTS='compress_qualpushdown_complex plan_skip_scan cagg_rewrites' \
  cmake --build build-pg16-debug --target regresscheck-t
~~~

Expected: all selected parent-commit tests pass. Save command output in the SDD report.

- [ ] **Step 4: Commit**

No commit: build artifacts and SDD reports remain ignored.

### Task 2: Remove rebased expected-output trailing whitespace

**Files:**
- Modify: tsl/test/isolation/expected/decompress_recompress_relid.out.
- Test: scoped whitespace check against the 2.29.1 baseline.

**Interfaces:**
- Preserves the isolation-test output byte-for-byte except for the accidental final blank line.

- [ ] **Step 1: Capture the failing check**

Run:

~~~bash
git diff --check 2.29.1..HEAD -- tsl/test/isolation/expected/decompress_recompress_relid.out
~~~

Expected: reports the final blank line at EOF.

- [ ] **Step 2: Remove only the final blank line**

Use an `apply_patch` deletion at the end of
`tsl/test/isolation/expected/decompress_recompress_relid.out`; do not regenerate
or reorder the expected output.

- [ ] **Step 3: Prove the repair**

Run:

~~~bash
git diff --check 2.29.1..HEAD -- tsl/test/isolation/expected/decompress_recompress_relid.out
git diff --word-diff=porcelain 2.29.1..HEAD -- tsl/test/isolation/expected/decompress_recompress_relid.out
~~~

Expected: the whitespace check is silent and the only net difference attributable
to this task is deletion of the final blank line.

- [ ] **Step 4: Commit**

~~~bash
git add tsl/test/isolation/expected/decompress_recompress_relid.out
git commit -m "Tests: remove trailing expected-output whitespace" -m "Remove the accidental final blank line introduced by the rebase."
~~~

### Task 3: Make the ColumnarScan namespace collision-proof

**Files:**
- Create: tsl/test/t/006_citus_columnar_customscan.pl.
- Modify: tsl/test/t/CMakeLists.txt, tsl/src/nodes/columnar_scan/planner.c, tsl/src/nodes/columnar_scan/columnar_scan.c.
- Modify: all expected-output files under test/expected, tsl/test/expected, and tsl/test/shared/expected containing Custom Scan (ColumnarScan).
- Test: new TAP gate plus installcheck.

**Interfaces:**
- Consumes: PostgreSQL's global CustomScan registry and a Citus Columnar library preloaded before Timescale.
- Produces: TimescaleDBColumnarScan plans that deserialize to the Timescale method in parallel workers.

- [ ] **Step 1: Build a fresh Citus PG16 dylib out of tree**

Run:

~~~bash
CITUS_BUILD=$(mktemp -d /tmp/pgpro-citus-pg16.XXXXXX)
cd "$CITUS_BUILD"
env \
  PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config \
  CPPFLAGS='-I/opt/homebrew/opt/lz4/include -I/opt/homebrew/opt/zstd/include' \
  LDFLAGS='-L/opt/homebrew/opt/lz4/lib -L/opt/homebrew/opt/zstd/lib' \
  /Users/fenoman/GitProjects/pgpro_citus/configure --without-libcurl
make -C "$CITUS_BUILD/src/backend/columnar" citus_columnar.dylib
test -f "$CITUS_BUILD/src/backend/columnar/citus_columnar.dylib"
~~~

Expected: the artifact is under the temporary build directory and no install action occurs.

- [ ] **Step 2: Write the failing opt-in TAP test**

Add 006_citus_columnar_customscan.pl to PROVE_TEST_FILES. Skip only when CITUS_COLUMNAR_LIBDIR is absent or PG_VERSION_MAJOR is not 16. Construct the node manually so configuration precedes start:

~~~perl
my $node = TimescaleNode->new('citus_columnar_customscan');
$node->init;
$node->append_conf('postgresql.conf',
  "dynamic_library_path='$ENV{CITUS_COLUMNAR_LIBDIR}:$ENV{TS_LOADER_DIR}:$ENV{TS_CORE_DIR}:$ENV{TS_TSL_DIR}:\$libdir'");
$node->append_conf('postgresql.conf',
  "shared_preload_libraries='citus_columnar,timescaledb'");
$node->append_conf('postgresql.conf',
  "max_parallel_workers=8\nmax_parallel_workers_per_gather=4");
$node->start;
$node->safe_psql('postgres', 'CREATE EXTENSION timescaledb');
~~~

The test creates sixteen compressed daily chunks with 5,000 rows each, forces a parallel aggregate, saves the serial result, and asserts a successful parallel result exactly equal to 80000|200040000. It also asserts EXPLAIN VERBOSE contains TimescaleDBColumnarScan and launched-worker evidence.

- [ ] **Step 3: Run the test-only change and record red**

Run:

~~~bash
cmake --build build-pg16-debug -j2 --target timescaledb timescaledb-tsl
CITUS_COLUMNAR_LIBDIR="$CITUS_BUILD/src/backend/columnar" \
TS_LOADER_DIR="$PWD/build-pg16-debug/src/loader" \
TS_CORE_DIR="$PWD/build-pg16-debug/src" \
TS_TSL_DIR="$PWD/build-pg16-debug/tsl/src" \
PROVE_TESTS='t/006_citus_columnar_customscan.pl' \
  cmake --build build-pg16-debug --target provecheck-t
~~~

Expected: FAIL on the old name, normally with wrong type of slot in a parallel worker or an old-name assertion failure.

- [ ] **Step 4: Rename both registry keys**

Set both method descriptors exactly as follows:

~~~c
static CustomScanMethods columnar_scan_plan_methods = {
    .CustomName = "TimescaleDBColumnarScan",
    .CreateCustomScanState = tsl_columnar_scan_state_create,
};

static CustomPathMethods columnar_scan_path_methods = {
    .CustomName = "TimescaleDBColumnarScan",
    .PlanCustomPath = columnar_scan_plan_create,
};
~~~

Do not change pointer-based plan classification or Citus source.

- [ ] **Step 5: Update rendered plan output mechanically**

Run:

~~~bash
rg -l -F 'Custom Scan (ColumnarScan)' \
  test/expected tsl/test/expected tsl/test/shared/expected \
  | xargs perl -pi -e 's/Custom Scan \(ColumnarScan\)/Custom Scan (TimescaleDBColumnarScan)/g'
~~~

Expected: pre-change scope is 122 files. Do not rewrite comments or C type names.

- [ ] **Step 6: Run green coverage**

Run:

~~~bash
cmake --build build-pg16-debug -j2 --target timescaledb timescaledb-tsl
CITUS_COLUMNAR_LIBDIR="$CITUS_BUILD/src/backend/columnar" \
TS_LOADER_DIR="$PWD/build-pg16-debug/src/loader" \
TS_CORE_DIR="$PWD/build-pg16-debug/src" \
TS_TSL_DIR="$PWD/build-pg16-debug/tsl/src" \
PROVE_TESTS='t/006_citus_columnar_customscan.pl' \
  cmake --build build-pg16-debug --target provecheck-t
cmake --build build-pg16-debug --target installcheck
git diff --check
~~~

Expected: Citus serial/parallel results match and no rendered old CustomScan name remains.

- [ ] **Step 7: Commit**

~~~bash
git add tsl/src/nodes/columnar_scan/planner.c \
  tsl/src/nodes/columnar_scan/columnar_scan.c \
  tsl/test/t/CMakeLists.txt tsl/test/t/006_citus_columnar_customscan.pl \
  test/expected tsl/test/expected tsl/test/shared/expected
git commit -m "Fix: namespace Timescale columnar custom scan"
~~~

### Task 4: Prevent lossy min/max pushdown beneath NullTest

**Files:**
- Create: .unreleased/fix_isnull_minmax_pushdown.
- Modify: tsl/src/nodes/columnar_scan/qual_pushdown.c.
- Modify: tsl/test/sql/compress_qualpushdown_complex.sql and its expected output.
- Test: compress_qualpushdown_complex.

**Interfaces:**
- Consumes: QualPushdownContext.can_pushdown and needs_recheck from child recursion.
- Produces: an unmodified outer transform when a child is only lossy, while exact range pushdown remains available.

- [ ] **Step 1: Add the red sparse NULL regression**

Append a compressed sparse_isnull hypertable with rows (2024-01-01, NULL) and (2024-01-02, 1). Add:

~~~sql
EXPLAIN (buffers off, costs off)
SELECT * FROM sparse_isnull WHERE (v > 0) IS NULL;
SELECT * FROM sparse_isnull WHERE (v > 0) IS NULL;

EXPLAIN (buffers off, costs off)
SELECT * FROM sparse_isnull WHERE (v > 0) IS NOT NULL;
SELECT * FROM sparse_isnull WHERE (v > 0) IS NOT NULL;
~~~

The intended output filters above TimescaleDBColumnarScan, returns the NULL row for IS NULL and 1 for IS NOT NULL, and retains ordinary v > 0 min/max pushdown.

- [ ] **Step 2: Run and record red**

Run:

~~~bash
cmake --build build-pg16-debug -j2 --target timescaledb timescaledb-tsl
TESTS=compress_qualpushdown_complex \
  cmake --build build-pg16-debug --target regresscheck-t
~~~

Expected: parent code loses the NULL row or shows the lossy compressed filter.

- [ ] **Step 3: Mutate transforming nodes through an isolated context**

In the transforming-node switch in qual_pushdown_mutator(), use:

~~~c
QualPushdownContext tmp_context = copy_context(context);
Node *pushed_down =
    expression_tree_mutator((Node *) orig_node, qual_pushdown_mutator, &tmp_context);
if (!tmp_context.can_pushdown || tmp_context.needs_recheck)
{
    context->can_pushdown = false;
    return orig_node;
}
return pushed_down;
~~~

Keep node tags and unrelated branches untouched. Add the unreleased text Fix wrong results with IS NULL and minmax pushdown.

- [ ] **Step 4: Verify green**

Run:

~~~bash
cmake --build build-pg16-debug -j2 --target timescaledb timescaledb-tsl
TESTS=compress_qualpushdown_complex \
  cmake --build build-pg16-debug --target regresscheck-t
git diff --check
~~~

Expected: both NULL truth tables are correct and ordinary v > 0 pushdown remains.

- [ ] **Step 5: Commit**

~~~bash
git add .unreleased/fix_isnull_minmax_pushdown \
  tsl/src/nodes/columnar_scan/qual_pushdown.c \
  tsl/test/sql/compress_qualpushdown_complex.sql \
  tsl/test/expected/compress_qualpushdown_complex.out
git commit -m "Fix: prevent lossy minmax pushdown beneath NullTest"
~~~

### Task 5: Make runtime ModifyHypertable EXPLAIN repeatable

**Files:**
- Modify: src/nodes/modify_hypertable.h and src/nodes/modify_hypertable.c.
- Modify: test/src/test_utils.c, test/sql/test_utils.sql, test/expected/test_utils.out.
- Test: test_utils and delete.

**Interfaces:**
- Consumes: ModifyHypertableState, SharedCounters, ExecutorRun_hook, and ExplainPrintPlan.
- Produces: one targetlist hide/restore cycle per executor and counter totals invariant under repeated polls.

- [ ] **Step 1: Add a separate runtime-EXPLAIN test hook**

Leave the existing ExecutorStart hook untouched. Add ExecutorRun enable/disable functions. After chaining to the previous or standard run, before ExecutorEnd, call:

~~~c
ExplainState *es = NewExplainState();
es->verbose = true;
ExplainPrintPlan(es, query_desc);
ExplainPrintPlan(es, query_desc);
~~~

For DELETE, inspect ModifyHypertableState after the second call and raise ERROR if the saved targetlist changed to NULL. For INSERT, set state->ctr->counters->batches_scanned to 1 after execution, invoke EXPLAIN twice, and raise ERROR unless state->batches_scanned is 1. SQL creates a two-chunk hypertable, performs both operations while enabled, and disables the hook.

- [ ] **Step 2: Execute the test-only change and record red**

Run:

~~~bash
cmake --build build-pg16-debug -j2 --target timescaledb
TESTS=test_utils cmake --build build-pg16-debug --target regresscheck
~~~

Expected: the old code fails the targetlist assertion and the controlled counter test.

- [ ] **Step 3: Add idempotent state fields and delta accounting**

Append these fields to the end of ModifyHypertableState only:

~~~c
bool explain_targetlists_hidden;
SharedCounters explain_last_counters;
~~~

Guard targetlist nullification with the boolean, set it after saving pointers, and restore based on the boolean rather than pointer truthiness. For every field currently read from state->ctr->counters, add the difference from explain_last_counters and then snapshot the current SharedCounters. Direct assignment is forbidden because MERGE can accumulate state statistics independently.

- [ ] **Step 4: Verify green**

Run:

~~~bash
cmake --build build-pg16-debug -j2 --target timescaledb
TESTS='test_utils delete' cmake --build build-pg16-debug --target regresscheck
git diff --check
~~~

Expected: repeated runtime EXPLAIN, the earlier NULL-ctr ExecutorStart regression, and prepared DELETE execution all pass.

- [ ] **Step 5: Commit**

~~~bash
git add src/nodes/modify_hypertable.h src/nodes/modify_hypertable.c \
  test/src/test_utils.c test/sql/test_utils.sql test/expected/test_utils.out
git commit -m "Fix: make ModifyHypertable EXPLAIN idempotent"
~~~

### Task 6: Correct the compressed SkipScan offset estimate

**Files:**
- Modify: tsl/src/nodes/skip_scan/planner.c and tsl/test/sql/plan_skip_scan.sql.in.
- Modify: generated plan_skip_scan expected files for supported majors.
- Test: plan_skip_scan.

**Interfaces:**
- Consumes: selectivity of clauses_needing_scan in the compressed SkipScan path.
- Produces: a non-negative limit offset equal to the expected preceding rejected tuples.

- [ ] **Step 1: Add a cost-visible compressed SkipScan case**

At the compressed skip_scan_htcl section of plan_skip_scan.sql.in, add this query with sequential scan disabled only around it:

~~~sql
EXPLAIN (buffers off, timing off, summary off)
SELECT DISTINCT ON (dev) dev, time
FROM skip_scan_htcl
WHERE time > 5
ORDER BY dev, time DESC;
~~~

Keep parallel disabled. Expected output must include costs and the compressed TimescaleDBColumnarScan/SkipScan path. Encode the intended result for OFFSET = max(0, floor(1 / selectivity) - 1).

- [ ] **Step 2: Run test-only change and record red**

Run:

~~~bash
cmake --build build-pg16-debug -j2 --target timescaledb timescaledb-tsl
TESTS=plan_skip_scan cmake --build build-pg16-debug --target regresscheck-t
~~~

Expected: only the new cost-visible output differs because old code adds one extra tuple per group.

- [ ] **Step 3: Correct the formula**

Change only the compressed-path offset calculation to:

~~~c
offset_until_qual_pass = Max(0, floor(1 / qual_selectivity) - 1);
~~~

Leave the selectivity floor, adjust_limit_rows_costs(), normal IndexPath branch, and multiplier unchanged.

- [ ] **Step 4: Verify green on all configured majors**

Run:

~~~bash
TESTS=plan_skip_scan cmake --build build-pg16-debug --target regresscheck-t
TESTS=plan_skip_scan cmake --build build-pg17-debug --target regresscheck-t
TESTS=plan_skip_scan cmake --build build-pg18-debug --target regresscheck-t
git diff --check
~~~

Expected: results and non-compressed paths are unchanged; only intended cost output changes.

- [ ] **Step 5: Commit**

~~~bash
git add tsl/src/nodes/skip_scan/planner.c tsl/test/sql/plan_skip_scan.sql.in \
  tsl/test/expected/plan_skip_scan-16.out \
  tsl/test/expected/plan_skip_scan-17.out \
  tsl/test/expected/plan_skip_scan-18.out \
  tsl/test/expected/plan_skip_scan-19.out
git commit -m "Fix: correct compressed SkipScan offset cost"
~~~

### Task 7: Bypass disabled CAGG rewrite traversal

**Files:**
- Modify: tsl/src/continuous_aggs/rewrite_with_caggs.c and tsl/test/sql/cagg_rewrites.sql.in.
- Modify: generated cagg_rewrites expected files.
- Test: cagg_rewrites plus 25 warm planning samples.

**Interfaces:**
- Consumes: ts_guc_enable_cagg_rewrites and ts_guc_cagg_rewrites_debug_info.
- Produces: parse unchanged and no CAGG tree walk when both controls are false.

- [ ] **Step 1: Add disabled-GUC semantic coverage**

After both CAGG GUCs are set to zero in cagg_rewrites.sql.in, add:

~~~sql
EXPLAIN (buffers off, costs off, timing off, summary off)
SELECT time_bucket(INTERVAL '2 day', day) AS bucket, count(device_id)
FROM conditions
GROUP BY bucket
ORDER BY 1
LIMIT 3;
~~~

Expected: raw conditions is planned, not a continuous aggregate. Existing enabled/debug cases protect both non-bypass modes.

- [ ] **Step 2: Record the performance red baseline**

With both GUCs disabled, capture 25 warm Planning Time samples of:

~~~sql
DISCARD PLANS;
EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON)
SELECT count(*)
FROM (SELECT g AS id FROM generate_series(1, 200) AS g) AS a
JOIN (SELECT g AS id FROM generate_series(1, 200) AS g) AS b USING (id);
~~~

Record median, settings, and plan text in the SDD report.

- [ ] **Step 3: Add the exact fast return**

At the top of continuous_agg_apply_rewrites(), add:

~~~c
if (!ts_guc_enable_cagg_rewrites && !ts_guc_cagg_rewrites_debug_info)
{
    return parse;
}
~~~

Do not alter the planner hook or enabled/debug branches.

- [ ] **Step 4: Verify the semantic guard and admission gate**

Run:

~~~bash
cmake --build build-pg16-debug -j2 --target timescaledb timescaledb-tsl
TESTS=cagg_rewrites cmake --build build-pg16-debug --target regresscheck-t
git diff --check
~~~

Repeat the 25 samples. Admit this commit only if the median improves at least 10%, all CAGG result-diff checks pass, and the disabled plan remains non-rewritten.

- [ ] **Step 5: Commit or defer**

If admitted:

~~~bash
git add tsl/src/continuous_aggs/rewrite_with_caggs.c \
  tsl/test/sql/cagg_rewrites.sql.in tsl/test/expected/cagg_rewrites-*.out
git commit -m "Perf: skip disabled CAGG rewrite traversal"
~~~

If the gate fails, revert this task's uncommitted changes, record both medians in the SDD ledger, and make no source commit.

### Task 8: Qualify ordinary database-wide VACUUM short-circuit

**Files:**
- Modify only after admission: src/process_utility.c, test/sql/vacuum.sql, test/expected/vacuum.out.
- Test: vacuum and two disposable-database benchmark profiles.

**Interfaces:**
- Consumes: VacuumStmt, the HYPERTABLE catalog, and the existing ACL logic.
- Produces: DDL_CONTINUE before Timescale's full pg_class enumeration when no permitted Timescale target exists.

- [ ] **Step 1: Add the semantic guard before production code**

After hypertable cleanup in vacuum.sql, add an ordinary table, insert/analyze rows, execute database-wide VACUUM ANALYZE, and query pg_stat_all_tables.n_live_tup. Add an ACL case if the fixture can run it: an inaccessible hypertable must not trigger interception for that role.

- [ ] **Step 2: Measure both profiles**

In a disposable database, create 1,000 ordinary tables and no hypertables; warm up then run ANALYZE eleven times and record median. Repeat with one hypertable the active user may analyze. Keep plan/result evidence for the ordinary semantic guard.

- [ ] **Step 3: Decide admission from the data**

Proceed only if the no-hypertable profile demonstrates a meaningful duplicate catalog traversal and the permitted-hypertable profile is not already materially regressed. If not admitted, revert test edits, record medians and leave no source commit.

- [ ] **Step 4: Implement the bounded preflight if admitted**

Add a static helper that scans HYPERTABLE with ScanIterator, resolves candidate relids, and calls the same vacuum_is_permitted_for_relation_compat() ACL check as ts_get_all_vacuum_rels(). Stop at the first permitted target. In process_vacuum(), only for stmt->rels == NIL, return DDL_CONTINUE before ts_get_all_vacuum_rels() if the helper finds none. Retain deep-copy, PG_TRY, compressed-chunk, and rebuild behavior.

- [ ] **Step 5: Verify and commit if admitted**

Run:

~~~bash
cmake --build build-pg16-debug -j2 --target timescaledb
TESTS=vacuum cmake --build build-pg16-debug --target regresscheck
git diff --check
~~~

Repeat both profiles. Commit only if no-hypertable improves and permitted-hypertable regresses no more than 3%:

~~~bash
git add src/process_utility.c test/sql/vacuum.sql test/expected/vacuum.out
git commit -m "Perf: bypass Timescale vacuum scan without hypertables"
~~~

### Task 9: Qualify planner relation-classification reuse

**Files:**
- Modify only after admission: src/planner/planner.c, src/planner/planner.h, test/sql/baserel_cache.sql, test/expected/baserel_cache.out.
- Test: baserel_cache, compression DML smoke, 10/100/1000-RTE profiles.

**Interfaces:**
- Consumes: preprocess relation lookup, BaserelInfo_insert(), get_or_add_baserel_from_cache(), and ts_classify_relation().
- Produces: a query-lifetime cached positive or negative classification without altering ONLY chunk marking.

- [ ] **Step 1: Add semantic coverage before planner code**

Extend baserel_cache.sql with a plain-table join, normal hypertable expansion, SELECT from ONLY a chunk, and ordinary UPDATE/DELETE. Keep the existing error/re-entry case first and assert both result and relevant plan shape.

- [ ] **Step 2: Record quantitative baseline**

For 10, 100, and 1,000 plain RTE joins and one mixed hypertable/plain query, run eleven DISCARD PLANS plus EXPLAIN ANALYZE TIMING OFF SUMMARY ON samples. Record medians and plan text. Admit only if ordinary 100/1000-RTE medians improve at least 10% with identical result/plan semantics.

- [ ] **Step 3: Implement only narrow preprocess seeding if admitted**

When preprocess sees a non-hypertable RTE_RELATION at planner.c lines 471-492, seed a BaserelInfoEntry with the chunk lookup result, including negative entry->ht == NULL. Make get_or_add_baserel_from_cache() reuse it. Preserve rte_mark_for_expansion() for rte->inh. Do not modify replace_modify_hypertable_paths() or direct compressed-chunk DML handling.

- [ ] **Step 4: Verify and commit if admitted**

Run:

~~~bash
cmake --build build-pg16-debug -j2 --target timescaledb
TESTS=baserel_cache cmake --build build-pg16-debug --target regresscheck
TESTS='compression_update_delete compression_dml' \
  cmake --build build-pg16-debug --target regresscheck-t
git diff --check
~~~

Repeat profiles and commit only on gate success:

~~~bash
git add src/planner/planner.c src/planner/planner.h \
  test/sql/baserel_cache.sql test/expected/baserel_cache.out
git commit -m "Perf: reuse planner relation classification"
~~~

If either gate fails, revert task-local edits, record the evidence, and make no source commit.

### Task 10: Record research-only dispositions for compression settings and sort cost

**Files:**
- Create: SDD workspace benchmark records only.
- Modify: none.
- Test: 10/100/1000 selected compressed chunks.

**Interfaces:**
- Consumes: ts_compression_settings_get() at build_compressioninfo() and the quadratic per-child cost in make_chunk_sorted_path().
- Produces: a follow-on design input, not a source change.

- [ ] **Step 1: Measure compression-settings density**

Compare queries touching 10, 100, and 1,000 compressed chunks with both matching settings and 10,000 unrelated settings. Record planning time and catalog scan count. Retain indexed point lookups unless a selective multi-key alternative wins both sparse and dense profiles.

- [ ] **Step 2: Measure sort-cost behavior**

For 10, 100, and 1,000 compressed chunks, capture EXPLAIN FORMAT JSON, selected plan, total cost, and execution time for ordered LIMIT queries whose sort cannot fully push below decompression. Reject any candidate that moves ordinary/non-columnar plans or slows the target by more than 3%.

- [ ] **Step 3: Record disposition**

Record evidence and disposition in the SDD ledger. This plan does not alter bulk compression-settings APIs, sort cost, or first/last aggregate grouping.

### Task 11: Run final multi-version and Citus validation

**Files:**
- Modify: none unless an accepted task's test exposes a defect in that task.
- Test: available installcheck, isolation, TAP, and PG16 Citus gate.

**Interfaces:**
- Consumes: accepted commits from Tasks 2-9 and the fresh Citus build.
- Produces: exact pass/fail/blocked evidence for final review.

- [ ] **Step 1: Configure PG17 and PG18 Debug builds**

Run:

~~~bash
cmake -S . -B build-pg17-debug \
  -DPG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config \
  -DPG_SOURCE_DIR=/tmp/timescaledb-testdeps-20260810/pg-src/postgresql-17.10 \
  -DCMAKE_BUILD_TYPE=Debug -DASSERTIONS=ON \
  -DREGRESS_CHECKS=ON -DREQUIRE_ALL_TESTS=ON
cmake -S . -B build-pg18-debug \
  -DPG_CONFIG=/opt/homebrew/opt/postgresql@18/bin/pg_config \
  -DPG_SOURCE_DIR=/tmp/timescaledb-testdeps-20260810/pg-src/postgresql-18.4 \
  -DCMAKE_BUILD_TYPE=Debug -DASSERTIONS=ON \
  -DREGRESS_CHECKS=ON -DREQUIRE_ALL_TESTS=ON
cmake --build build-pg17-debug -j2
cmake --build build-pg18-debug -j2
~~~

- [ ] **Step 2: Run every locally available installed suite**

Run:

~~~bash
cmake --build build-pg16-debug --target installcheck
cmake --build build-pg17-debug --target installcheck
cmake --build build-pg18-debug --target installcheck
~~~

Expected: every available target passes; missing TAP prerequisites are named exactly.

- [ ] **Step 3: Re-run the PG16 Citus gate**

Re-run Task 3 Step 6 with fresh CITUS_COLUMNAR_LIBDIR. Preserve serial/parallel equality and a plan that names TimescaleDBColumnarScan.

- [ ] **Step 4: Final hygiene and branch review**

Run:

~~~bash
git diff --check 38f6f7e62..HEAD
git status --short
git log --oneline 38f6f7e62..HEAD
~~~

Expected: one commit per accepted behavioral change, no Citus source checkout edits, and a complete SDD review package plus ledger decisions.
