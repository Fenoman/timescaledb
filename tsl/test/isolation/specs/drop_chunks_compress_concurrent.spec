# This file and its contents are licensed under the Timescale License.
# Please see the included NOTICE for copyright information and
# LICENSE-TIMESCALE for a copy of the license.

# drop_chunks() against a compress_chunk() that is already working on the
# same chunk. compress_chunk() holds an ExclusiveLock on the chunk while
# it compresses and updates the chunk catalog row only at the end.
# drop_chunks() has to take its lock on the chunk relation before it locks
# the catalog row, otherwise the two wait for each other and one of them
# fails with a deadlock. The waitpoint stops compress_chunk() right after
# it has taken its locks.

setup {
	CREATE TABLE conditions (time timestamptz NOT NULL, device int, temp float);
	SELECT create_hypertable('conditions', 'time', chunk_time_interval => interval '1 day');
	INSERT INTO conditions
	SELECT t, 1, random() * 100
	FROM generate_series('2018-12-01 00:00'::timestamptz, '2018-12-02 23:00', '1 hour') t;
	ALTER TABLE conditions SET (timescaledb.compress, timescaledb.compress_segmentby = 'device');
}

teardown {
	DROP TABLE conditions;
}

session "s1"
setup {
	SET SESSION lock_timeout = '10s';
	SET SESSION deadlock_timeout = '500ms';
}
step "s1_compress" {
	SELECT count(compress_chunk(chunk))
	FROM (SELECT chunk FROM show_chunks('conditions') chunk ORDER BY 1 LIMIT 1) AS c;
}

session "s2"
setup {
	SET SESSION lock_timeout = '10s';
	SET SESSION deadlock_timeout = '500ms';
}
step "s2_drop" {
	SELECT count(*) FROM drop_chunks('conditions', older_than => '2018-12-03 00:00'::timestamptz);
}
step "s2_chunks" {
	SELECT count(*) FROM show_chunks('conditions');
}

session "s3"
step "s3_enable" {
	SELECT debug_waitpoint_enable('compress_chunk_impl_start');
}
step "s3_release" {
	SELECT debug_waitpoint_release('compress_chunk_impl_start');
}

permutation "s3_enable" "s1_compress" "s2_drop" "s3_release" "s2_chunks"
