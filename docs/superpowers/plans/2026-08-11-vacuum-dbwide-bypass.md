# VACUUM Database-Wide Bypass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop intercepting non-FULL database-wide VACUUM/ANALYZE in TimescaleDB so the daily maintenance statement runs through PostgreSQL's vanilla utility path with zero Timescale overhead.

**Architecture:** One early `DDL_CONTINUE` in `process_vacuum()` for `stmt->rels == NIL && !ctx.is_vacuumfull`, guarded by a new solo regression test that pins "chunks and ordinary tables both get statistics from a database-wide statement" on both the parent and the patched code. Admission is an A/B/A/B benchmark on a 5000-table + 1-hypertable database, recorded under `docs/superpowers/benchmarks/`.

**Tech Stack:** C (PostgreSQL utility hook), pg_regress via CMake targets, Homebrew PostgreSQL 16/17/18, Debug builds in `build_rebase_pg{16,17,18}_debug`.

## Global Constraints

- Branch: `local_2.29.1`; base commit `65db420f5` or later. No push.
- Spec: `docs/superpowers/specs/2026-08-11-vacuum-dbwide-bypass-design.md`.
- One source commit for code + tests; benchmark record is a separate docs commit (audit-series convention).
- Commit messages: subject ≤ 50 chars, body wrapped at 72 (repo hook enforces).
- All `regresscheck` runs need `LC_ALL=ru_RU.UTF-8 LANG=ru_RU.UTF-8` (macOS postmaster multithread guard); a single spurious `postmaster failed` is retried once.
- `VACUUM FULL` database-wide, explicit-relation VACUUM, and `ONLY_DATABASE_STATS` behavior must not change.
- Benchmark gate: reproducible median improvement in both A/B pairs and byte-identical semantic-guard test output; on gate failure record numbers and do not commit the source change.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/process_utility.c` | The two-line bypass in `process_vacuum()` |
| `test/sql/vacuum_dbwide.sql` (new) | Solo semantic guard: db-wide ANALYZE / VACUUM (ANALYZE) reach ordinary tables and chunks |
| `test/expected/vacuum_dbwide.out` (new) | Expected output, identical before and after the code change |
| `test/sql/CMakeLists.txt` | Register the new test file and add it to `SOLO_TESTS` |
| `docs/superpowers/benchmarks/2026-08-11-vacuum-dbwide-bypass.md` (new) | A/B/A/B measurement record and admission decision |

### Task 1: Semantic guard test, green on parent

**Files:**
- Create: `test/sql/vacuum_dbwide.sql`
- Create: `test/expected/vacuum_dbwide.out`
- Modify: `test/sql/CMakeLists.txt` (TEST_FILES list, alphabetical position after `vacuum.sql`; SOLO_TESTS set)

**Interfaces:**
- Consumes: `create_hypertable()` SQL API; `pg_class.reltuples` (transactional, no stats-collector timing races; `-1` for never-analyzed relations on PG ≥ 14).
- Produces: test name `vacuum_dbwide` used by every later validation step.

- [ ] **Step 1: Write the test file**

`test/sql/vacuum_dbwide.sql` (new; solo because a database-wide VACUUM inside a parallel group can collide with concurrent DDL of neighboring tests):

```sql
-- This file and its contents are licensed under the Apache License 2.0.
-- Please see the included NOTICE for copyright information and
-- LICENSE-APACHE for a copy of the license.

-- Database-wide VACUUM/ANALYZE (no relation list, no FULL) is not
-- intercepted by Timescale: chunks and compressed chunks are ordinary
-- pg_class entries, so PostgreSQL's own enumeration must reach both
-- ordinary tables and chunks. reltuples is used as the witness because
-- VACUUM/ANALYZE update it transactionally in the catalog.
CREATE TABLE vacuum_dbwide_plain(a int);
INSERT INTO vacuum_dbwide_plain SELECT generate_series(1, 100);
CREATE TABLE vacuum_dbwide_ht(time timestamptz NOT NULL, v int);
SELECT create_hypertable('vacuum_dbwide_ht', 'time',
       chunk_time_interval => interval '1 day');
INSERT INTO vacuum_dbwide_ht VALUES ('2026-01-01', 1), ('2026-01-02', 2);

-- Never analyzed yet: reltuples is -1
SELECT reltuples::int AS plain_before FROM pg_class
WHERE relname = 'vacuum_dbwide_plain';

ANALYZE;

SELECT reltuples::int AS plain_after_analyze FROM pg_class
WHERE relname = 'vacuum_dbwide_plain';
SELECT sum(c.reltuples)::int AS chunks_after_analyze
FROM pg_class c
JOIN pg_inherits i ON c.oid = i.inhrelid
WHERE i.inhparent = 'vacuum_dbwide_ht'::regclass;

-- Exercise the VACUUM flavor of the database-wide path too
VACUUM (ANALYZE);

SELECT reltuples::int AS plain_after_vacuum FROM pg_class
WHERE relname = 'vacuum_dbwide_plain';
SELECT sum(c.reltuples)::int AS chunks_after_vacuum
FROM pg_class c
JOIN pg_inherits i ON c.oid = i.inhrelid
WHERE i.inhparent = 'vacuum_dbwide_ht'::regclass;

DROP TABLE vacuum_dbwide_plain;
DROP TABLE vacuum_dbwide_ht;
```

- [ ] **Step 2: Register the test as solo**

In `test/sql/CMakeLists.txt`:
- add `vacuum_dbwide.sql` to the main `TEST_FILES` list directly after the
  `vacuum.sql` line;
- add `vacuum_dbwide` to `set(SOLO_TESTS ...)` (name without `.sql`,
  alphabetical placement inside the set is fine).

- [ ] **Step 3: First run to capture output (expected missing → diff)**

```bash
cd /Users/fenoman/GitProjects/timescaledb
touch test/expected/vacuum_dbwide.out
LC_ALL=ru_RU.UTF-8 LANG=ru_RU.UTF-8 TESTS='vacuum_dbwide' \
  cmake --build build_rebase_pg16_debug --target regresscheck
```

Expected: FAIL (empty expected vs real output). Inspect
`build_rebase_pg16_debug/test/results/vacuum_dbwide.out`: `plain_before`
must be `-1`, `plain_after_analyze` `100`, `chunks_after_analyze` `2`,
`plain_after_vacuum` `100`, `chunks_after_vacuum` `2`. Any other values —
stop and investigate before proceeding.

- [ ] **Step 4: Install the verified output as expected**

```bash
cp build_rebase_pg16_debug/test/results/vacuum_dbwide.out \
   test/expected/vacuum_dbwide.out
```

- [ ] **Step 5: Re-run to verify green on the parent code**

```bash
LC_ALL=ru_RU.UTF-8 LANG=ru_RU.UTF-8 TESTS='vacuum_dbwide' \
  cmake --build build_rebase_pg16_debug --target regresscheck
```

Expected: `ok 1 - vacuum_dbwide`. This green-on-parent run is the semantic
baseline; the red gate for this change is the benchmark (Task 3), per the
Wave-2 convention. No commit yet — code and tests land together in Task 4.

### Task 2: The bypass

**Files:**
- Modify: `src/process_utility.c` (in `process_vacuum()`, immediately after the options `foreach` loop, before `if (stmt->rels == NIL)`)

**Interfaces:**
- Consumes: `ctx.is_vacuumfull` computed by the options loop above; `DDL_CONTINUE` return contract of process_utility handlers.
- Produces: the patch file `vacuum-bypass.patch` used by Task 3 to toggle A/B states.

- [ ] **Step 1: Apply the two-line change**

Locate in `src/process_utility.c` (function `process_vacuum`, directly after
the closing brace of the `foreach (lc, stmt->options)` loop):

```c
	if (stmt->rels == NIL)
	{
		vacuum_rels = ts_get_all_vacuum_rels(is_vacuumcmd, &ctx);
```

Insert before that `if`:

```c
	/*
	 * A non-FULL database-wide statement needs no Timescale involvement:
	 * chunks and compressed chunks are ordinary pg_class entries, so
	 * PostgreSQL's own enumeration already covers them, and columnstore
	 * rebuild registration only applies to VACUUM FULL. Skip the
	 * duplicate pg_class scan and let the standard path execute it.
	 */
	if (stmt->rels == NIL && !ctx.is_vacuumfull)
	{
		return DDL_CONTINUE;
	}

```

- [ ] **Step 2: Save the toggle patch for the benchmark**

```bash
cd /Users/fenoman/GitProjects/timescaledb
git diff src/process_utility.c > \
  "$SCRATCHPAD/vacuum-bypass.patch"
```

(`$SCRATCHPAD` = the session scratchpad directory; any stable absolute path
works.)

- [ ] **Step 3: Build, install, run the guard plus neighbors**

```bash
make -C build_rebase_pg16_debug -j8 && make -C build_rebase_pg16_debug install
LC_ALL=ru_RU.UTF-8 LANG=ru_RU.UTF-8 TESTS='vacuum vacuum_dbwide vacuum_parallel' \
  cmake --build build_rebase_pg16_debug --target regresscheck
```

Expected: all selected tests `ok`, with `vacuum_dbwide` output byte-identical
to the parent run (that is the semantic guard). No commit yet.

### Task 3: A/B/A/B benchmark and admission gate

**Files:**
- Create: `docs/superpowers/benchmarks/2026-08-11-vacuum-dbwide-bypass.md`

**Interfaces:**
- Consumes: `vacuum-bypass.patch` from Task 2; Homebrew PG16 binaries at `/opt/homebrew/opt/postgresql@16/bin`; the CAGG record `docs/superpowers/benchmarks/2026-08-11-cagg-rewrites-disabled-early-exit.md` as the protocol template.
- Produces: recorded medians and the admission decision consumed by Task 4.

- [ ] **Step 1: Create the dedicated benchmark cluster**

```bash
B=/opt/homebrew/opt/postgresql@16/bin
D="$SCRATCHPAD/vacbench-pgdata"
LC_ALL=ru_RU.UTF-8 LANG=ru_RU.UTF-8 "$B/initdb" -D "$D" --auth=trust >/dev/null
cat >> "$D/postgresql.conf" <<'EOF'
port = 55436
shared_preload_libraries = 'timescaledb'
timescaledb.telemetry_level = off
autovacuum = off
EOF
LC_ALL=ru_RU.UTF-8 LANG=ru_RU.UTF-8 "$B/pg_ctl" -D "$D" -l "$D/log" -w start
"$B/createdb" -p 55436 vacbench
"$B/psql" -p 55436 -d vacbench -c "CREATE EXTENSION timescaledb"
```

- [ ] **Step 2: Populate 5000 ordinary tables + 1 hypertable**

```bash
"$B/psql" -p 55436 -d vacbench <<'EOF'
SELECT format('CREATE TABLE bench_t%s(a int)', g)
FROM generate_series(1, 5000) g \gexec
CREATE TABLE bench_ht(time timestamptz NOT NULL, v int);
SELECT create_hypertable('bench_ht', 'time',
       chunk_time_interval => interval '1 day');
INSERT INTO bench_ht
SELECT '2026-01-01'::timestamptz + (g || ' hours')::interval, g
FROM generate_series(1, 240) g;
EOF
```

- [ ] **Step 3: Write the sampling script**

`$SCRATCHPAD/vacbench-run.sh`:

```bash
#!/bin/bash
# 30 timed database-wide ANALYZE in one warm session; caller drops first 5.
B=/opt/homebrew/opt/postgresql@16/bin
"$B/psql" -p 55436 -d vacbench -q -c '\timing on' \
  -f <(for i in $(seq 1 30); do echo "ANALYZE;"; done) 2>&1 |
  grep -E '^Time:' | awk '{print $2}'
```

- [ ] **Step 4: Run the four batches**

State toggling between batches (rebuild + reinstall + cluster restart each
time):

```bash
# state OFF (baseline): patch reversed
git apply -R "$SCRATCHPAD/vacuum-bypass.patch"
make -C build_rebase_pg16_debug -j8 && make -C build_rebase_pg16_debug install
"$B/pg_ctl" -D "$D" -w restart -l "$D/log"
bash "$SCRATCHPAD/vacbench-run.sh" > "$SCRATCHPAD/before1.txt"

# state ON: patch applied
git apply "$SCRATCHPAD/vacuum-bypass.patch"
make -C build_rebase_pg16_debug -j8 && make -C build_rebase_pg16_debug install
"$B/pg_ctl" -D "$D" -w restart -l "$D/log"
bash "$SCRATCHPAD/vacbench-run.sh" > "$SCRATCHPAD/after1.txt"

# repeat both once more -> before2.txt, after2.txt
```

For each of the four files: drop the first 5 values, compute median, mean,
stdev, min, max over the remaining 25. Verify with `git diff
src/process_utility.c` before each batch that the tree is in the intended
state (patch present for `after*`, absent for `before*`).

- [ ] **Step 5: Gate decision and record**

Admission requires: `after1` median < `before1` median AND `after2` median <
`before2` median, with a stable absolute delta between pairs. Write
`docs/superpowers/benchmarks/2026-08-11-vacuum-dbwide-bypass.md` following
the structure of the CAGG record (History / Method / Results table / Gate /
Decision), including the exact cluster settings and all four batch
statistics. Stop the cluster:

```bash
"$B/pg_ctl" -D "$D" stop
```

If the gate fails: leave the tree with the patch REVERSED, still commit the
benchmark record (with the negative decision), skip Task 4 Step 1, and
report.

### Task 4: Commit

**Files:**
- Commit 1: `src/process_utility.c`, `test/sql/vacuum_dbwide.sql`, `test/expected/vacuum_dbwide.out`, `test/sql/CMakeLists.txt`
- Commit 2: `docs/superpowers/benchmarks/2026-08-11-vacuum-dbwide-bypass.md`

- [ ] **Step 1: Source commit (only if the gate passed)**

```bash
git add src/process_utility.c test/sql/vacuum_dbwide.sql \
  test/expected/vacuum_dbwide.out test/sql/CMakeLists.txt
git commit -m "Perf: bypass db-wide VACUUM interception" -m "A database-wide VACUUM/ANALYZE without FULL gains nothing from
Timescale's interception: chunks and compressed chunks are ordinary
pg_class entries, ctx.chunk_rels stays empty in this branch, and
columnstore rebuild registration only applies to VACUUM FULL. Return
DDL_CONTINUE early so the statement runs through PostgreSQL's own
utility path, removing a duplicate pg_class scan with per-tuple ACL
checks, hypertable-cache probes and two deep copies of the relation
list. VACUUM FULL and explicit-relation statements keep the current
interception. Guarded by the new solo vacuum_dbwide test, which pins
identical statistics behavior before and after the change."
```

- [ ] **Step 2: Benchmark record commit**

```bash
git add docs/superpowers/benchmarks/2026-08-11-vacuum-dbwide-bypass.md
git commit -m "Docs: record db-wide VACUUM bypass benchmark" -m "A/B/A/B medians for database-wide ANALYZE on a 5000-table plus one
hypertable database, protocol as in the CAGG early-exit record."
```

- [ ] **Step 3: Hygiene**

```bash
git status --short   # expected: clean
git diff --check 65db420f5..HEAD
```

Expected for the check: only trailing-whitespace complaints inside
`test/expected/vacuum_dbwide.out` (psql pads result-set header lines; the
same artifact exists in 274 expected files of the repository). Anything
outside that file is a real defect.

### Task 5: Multi-version validation

**Files:** none modified.

- [ ] **Step 1: PG17**

```bash
make -C build_rebase_pg17_debug -j8 && make -C build_rebase_pg17_debug install
LC_ALL=ru_RU.UTF-8 LANG=ru_RU.UTF-8 TESTS='vacuum vacuum_dbwide vacuum_parallel' \
  cmake --build build_rebase_pg17_debug --target regresscheck
```

Expected: all ok.

- [ ] **Step 2: PG18 (reconfigure first — cache has port 5432 and full TAP requirement)**

```bash
cd build_rebase_pg18_debug
cmake -DREQUIRE_ALL_TESTS=OFF -DTEST_PGPORT_LOCAL=5433 .
make -j8 && make install
LC_ALL=ru_RU.UTF-8 LANG=ru_RU.UTF-8 TESTS='vacuum vacuum_dbwide vacuum_parallel' \
  cmake --build . --target regresscheck
```

Expected: all ok. If the PG18 Homebrew installation is missing or the
build fails for environment reasons, record the exact blocker in the final
report instead of treating it as passing.

- [ ] **Step 3: Report**

Summarize per-version results, benchmark medians, and the gate decision.
