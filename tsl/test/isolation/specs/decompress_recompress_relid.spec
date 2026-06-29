# This file and its contents are licensed under the Timescale License.
# Please see the included NOTICE for copyright information and
# LICENSE-TIMESCALE for a copy of the license.

setup {
  CREATE TABLE i8_relid_race(time timestamptz NOT NULL, device int, v int);
  SELECT FROM create_hypertable('i8_relid_race', 'time', chunk_time_interval => INTERVAL '1 day');
  ALTER TABLE i8_relid_race SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time',
    timescaledb.compress_segmentby = 'device'
  );
  INSERT INTO i8_relid_race
  SELECT '2024-01-01'::timestamptz + (i || ' minutes')::interval, i % 3, i
  FROM generate_series(1, 100) i;
  SELECT FROM (SELECT compress_chunk(c) FROM show_chunks('i8_relid_race') c) q;
}

teardown {
  DROP TABLE i8_relid_race;
}

session "wp"
step "wp_before_lock_on" { SELECT debug_waitpoint_enable('decompress_chunk_impl_before_lock') IS NULL AS x; }
step "wp_before_lock_off" { SELECT count(*) AS n
  FROM pg_locks
  WHERE NOT granted AND locktype = 'advisory' AND mode = 'ShareLock';
  SELECT debug_waitpoint_release('decompress_chunk_impl_before_lock') IS NULL AS x;
}

session "s1"
setup {
  SET client_min_messages = ERROR;
}
step "s1_decompress" { SELECT count(*) FROM (SELECT decompress_chunk(c) FROM show_chunks('i8_relid_race') c) q;
  SELECT count(*) FROM i8_relid_race;
}

session "s2"
setup {
  SET client_min_messages = ERROR;
  SET timescaledb.enable_in_memory_recompression = on;
}
step "s2_recompress" { SELECT count(*) FROM (SELECT compress_chunk(c, recompress => true)
    FROM show_chunks('i8_relid_race') c) q;
}

permutation "wp_before_lock_on" "s1_decompress" "s2_recompress" "wp_before_lock_off"
