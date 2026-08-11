-- This file and its contents are licensed under the Timescale License.
-- Please see the included NOTICE for copyright information and
-- LICENSE-TIMESCALE for a copy of the license.

\c :TEST_DBNAME :ROLE_SUPERUSER

-- An extension may build its own ExplainState with analyze set and inspect
-- the plan right after ExecutorStart, before the executor has created any
-- instrumentation. The ColumnarScan explain callback must tolerate a NULL
-- instrument.
CREATE OR REPLACE FUNCTION enable_explain_analyze_in_executor_start() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_enable_explain_analyze_in_executor_start' LANGUAGE C VOLATILE;
CREATE OR REPLACE FUNCTION disable_explain_analyze_in_executor_start() RETURNS VOID
    AS :MODULE_PATHNAME, 'ts_test_disable_explain_analyze_in_executor_start' LANGUAGE C VOLATILE;

CREATE TABLE early_explain_columnar (time timestamptz NOT NULL, device int, value float);
SELECT create_hypertable('early_explain_columnar', 'time');
ALTER TABLE early_explain_columnar
    SET (timescaledb.compress, timescaledb.compress_segmentby = 'device');
INSERT INTO early_explain_columnar
    VALUES ('2026-01-01 00:00+00', 1, 1.5), ('2026-01-01 01:00+00', 1, 2.5);
SELECT count(compress_chunk(c)) FROM show_chunks('early_explain_columnar') c;

DO $$ BEGIN PERFORM enable_explain_analyze_in_executor_start(); END $$;
SELECT device, value FROM early_explain_columnar ORDER BY time;
DO $$ BEGIN PERFORM disable_explain_analyze_in_executor_start(); END $$;
-- End early ExecutorStart EXPLAIN ANALYZE test.
