-- This file and its contents are licensed under the Apache License 2.0.
-- Please see the included NOTICE for copyright information and
-- LICENSE-APACHE for a copy of the license.

-- Test that the baserel cache is not clobbered if there's an error
-- in a SQL function.
CREATE TABLE valid_ids
(
  id UUID PRIMARY KEY
);

CREATE FUNCTION DEFAULT_UUID(TEXT DEFAULT '') RETURNS UUID AS $$
  BEGIN
    RETURN COALESCE($1, '')::UUID;
  EXCEPTION WHEN invalid_text_representation THEN
    RETURN '00000000-0000-0000-0000-000000000000';
  END;
$$ LANGUAGE PLPGSQL IMMUTABLE;

CREATE FUNCTION KNOWN_ID(UUID, TEXT) RETURNS UUID AS $$
  SELECT COALESCE(
    (SELECT id FROM valid_ids WHERE id = $1),
    DEFAULT_UUID()
  );
$$ LANGUAGE SQL;

SELECT KNOWN_ID(NULL, ''), KNOWN_ID(NULL, '');

-- Classification through the baserel cache must give identical results
-- whether a relation is first seen during query preprocessing or during
-- path generation: plain relations (also repeated in one query),
-- hypertable expansion, ONLY on a hypertable, and direct chunk access.
CREATE TABLE brc_plain(id int PRIMARY KEY, v int);
INSERT INTO brc_plain SELECT g, g FROM generate_series(1, 100) g;
CREATE TABLE brc_ht(time timestamptz NOT NULL, v int);
DO $$ BEGIN PERFORM create_hypertable('brc_ht', 'time',
       chunk_time_interval => interval '1 day'); END $$;
INSERT INTO brc_ht VALUES ('2026-01-01', 1), ('2026-01-02', 2);
ANALYZE brc_plain, brc_ht;

-- repeated range table entries of the same plain table
SELECT count(*) FROM brc_plain a
JOIN brc_plain b USING (id)
JOIN brc_plain c USING (id);

-- plain table and hypertable expansion in one query
SELECT count(*) FROM brc_plain p, brc_ht h WHERE p.id = h.v;

-- ONLY on the hypertable root scans no chunks
SELECT count(*) FROM ONLY brc_ht;

-- direct chunk access, with and without ONLY
SELECT format('%I.%I', chunk_schema, chunk_name) AS chunk
FROM timescaledb_information.chunks
WHERE hypertable_name = 'brc_ht'
ORDER BY 1 LIMIT 1 \gset
SELECT count(*) FROM :chunk;
SELECT count(*) FROM ONLY :chunk;

DROP TABLE brc_plain;
DROP TABLE brc_ht;
