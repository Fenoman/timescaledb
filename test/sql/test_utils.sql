-- This file and its contents are licensed under the Apache License 2.0.
-- Please see the included NOTICE for copyright information and
-- LICENSE-APACHE for a copy of the license.

\c :TEST_DBNAME :ROLE_SUPERUSER
CREATE OR REPLACE FUNCTION test.condition() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_utils_condition' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
CREATE OR REPLACE FUNCTION test.int64_eq() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_utils_int64_eq' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
CREATE OR REPLACE FUNCTION test.ptr_eq() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_utils_ptr_eq' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
CREATE OR REPLACE FUNCTION test.double_eq() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_utils_double_eq' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
SET ROLE :ROLE_DEFAULT_PERM_USER;

-- We're testing that the test utils work and generate errors on
-- failing conditions
\set ON_ERROR_STOP 0
SELECT test.condition();
SELECT test.int64_eq();
SELECT test.ptr_eq();
SELECT test.double_eq();
\set ON_ERROR_STOP 1

-- Test debug points
--
\set ECHO all

\c :TEST_DBNAME :ROLE_SUPERUSER
-- debug point already enabled
SELECT debug_waitpoint_enable('test_debug_point');
\set ON_ERROR_STOP 0
SELECT debug_waitpoint_enable('test_debug_point');
\set ON_ERROR_STOP 1
SELECT debug_waitpoint_release('test_debug_point');

-- debug point not enabled
\set ON_ERROR_STOP 0
SELECT debug_waitpoint_release('test_debug_point');
\set ON_ERROR_STOP 1

-- error injections
--
CREATE OR REPLACE FUNCTION test_error_injection(TEXT)
RETURNS VOID
AS :MODULE_PATHNAME, 'ts_test_error_injection'
LANGUAGE C VOLATILE STRICT;
SET ROLE :ROLE_DEFAULT_PERM_USER;

SELECT test_error_injection('test_error');

SELECT debug_waitpoint_enable('test_error');
\set ON_ERROR_STOP 0
SELECT test_error_injection('test_error');
\set ON_ERROR_STOP 1

SELECT debug_waitpoint_release('test_error');
SELECT test_error_injection('test_error');

-- Test Scanner
RESET ROLE;
CREATE OR REPLACE FUNCTION test.scanner() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_scanner' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
SET ROLE :ROLE_DEFAULT_PERM_USER;

-- Create two chunks to scan in the test
CREATE TABLE hyper (time timestamptz, temp float);
SELECT create_hypertable('hyper', 'time');
INSERT INTO hyper VALUES ('2021-01-01', 1.0), ('2022-01-01', 2.0);
SELECT test.scanner();

-- Test errdata_to_jsonb
RESET ROLE;
CREATE OR REPLACE FUNCTION test.errdata_to_jsonb() RETURNS JSONB
AS :MODULE_PATHNAME, 'ts_test_errdata_to_jsonb' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
SELECT test.errdata_to_jsonb();

-- An extension may inspect the plan right after ExecutorStart. This must work
-- before ModifyHypertable initializes ChunkTupleRouting on the first executor
-- call.
CREATE OR REPLACE FUNCTION test.enable_explain_in_executor_start() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_enable_explain_in_executor_start' LANGUAGE C VOLATILE;
CREATE OR REPLACE FUNCTION test.disable_explain_in_executor_start() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_disable_explain_in_executor_start' LANGUAGE C VOLATILE;

CREATE TABLE early_explain_insert (time timestamptz NOT NULL, value integer);
DO $$ BEGIN PERFORM create_hypertable('early_explain_insert', 'time'); END $$;
DO $$ BEGIN PERFORM test.enable_explain_in_executor_start(); END $$;
INSERT INTO early_explain_insert VALUES ('2026-01-01', 1);
MERGE INTO early_explain_insert h
USING (VALUES ('2026-01-02'::timestamptz, 2)) AS s(t, v)
ON h.time = s.t
WHEN MATCHED THEN UPDATE SET value = s.v
WHEN NOT MATCHED THEN INSERT VALUES (s.t, s.v);
-- A CTE DELETE also exposes the executor-only Result before the first tuple.
CREATE TABLE early_explain_reference (value integer PRIMARY KEY);
INSERT INTO early_explain_reference VALUES (1);
BEGIN;
WITH cte AS (
    SELECT h.value FROM early_explain_insert h
    LEFT JOIN early_explain_reference r ON r.value = h.value
    WHERE r.value IS NULL
)
DELETE FROM early_explain_insert h USING cte c WHERE c.value = h.value;
DO $$ BEGIN
    IF (SELECT array_agg(value ORDER BY value) FROM early_explain_insert) IS DISTINCT FROM ARRAY[1] THEN
        RAISE EXCEPTION 'early EXPLAIN DELETE produced incorrect rows';
    END IF;
END $$;
ROLLBACK;
DO $$ BEGIN
    IF (SELECT count(*) FROM early_explain_insert) <> 2 THEN
        RAISE EXCEPTION 'early EXPLAIN DELETE rollback did not restore rows';
    END IF;
END $$;
UPDATE early_explain_insert SET value = 3 WHERE value = 2;
DO $$ BEGIN PERFORM test.disable_explain_in_executor_start(); END $$;
-- End early ExecutorStart EXPLAIN test.

-- An extension may inspect an executed plan repeatedly before ExecutorEnd.
-- Repeated EXPLAIN must preserve saved targetlists and must not double-count
-- shared ChunkTupleRouting counters.
CREATE OR REPLACE FUNCTION test.enable_explain_in_executor_run() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_enable_explain_in_executor_run' LANGUAGE C VOLATILE;
CREATE OR REPLACE FUNCTION test.disable_explain_in_executor_run() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_disable_explain_in_executor_run' LANGUAGE C VOLATILE;

CREATE TABLE runtime_explain (time timestamptz NOT NULL, value integer);
DO $$ BEGIN PERFORM create_hypertable('runtime_explain', 'time'); END $$;
INSERT INTO runtime_explain VALUES ('2021-01-01', 1), ('2022-01-01', 2);
DO $$ BEGIN PERFORM test.enable_explain_in_executor_run(); END $$;
\set ON_ERROR_STOP 0
DELETE FROM runtime_explain WHERE time > '2020-01-01'::text::timestamptz;
INSERT INTO runtime_explain VALUES ('2021-01-01', 3);
DO $$ BEGIN PERFORM test.disable_explain_in_executor_run(); END $$;
\set ON_ERROR_STOP 1
-- End runtime ExecutorRun EXPLAIN test.
