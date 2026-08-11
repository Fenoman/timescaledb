-- This file and its contents are licensed under the Apache License 2.0.
-- Please see the included NOTICE for copyright information and
-- LICENSE-APACHE for a copy of the license.

-- Run as superuser so a database-wide statement analyzes every relation
-- without permission warnings; their exact set depends on catalog
-- contents and on who enumerates relations, which is not what this test
-- pins down.
\c :TEST_DBNAME :ROLE_SUPERUSER

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
