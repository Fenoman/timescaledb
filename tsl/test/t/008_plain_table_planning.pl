# This file and its contents are licensed under the Timescale License.
# Please see the included NOTICE for copyright information and
# LICENSE-TIMESCALE for a copy of the license.

use strict;
use warnings;
use TimescaleNode;
use Test::More;

my $node = TimescaleNode->create('plain_table_planning');
$node->safe_psql('postgres', q[
	CREATE TABLE plain(time bigint NOT NULL, value integer);
	INSERT INTO plain VALUES (1, 10);
	CREATE TABLE ht(time bigint NOT NULL, value integer);
	SELECT create_hypertable('ht', 'time', chunk_time_interval => 10);
]);

for my $query ('SELECT * FROM plain',
	'INSERT INTO plain SELECT 2, 20 WHERE false',
	'UPDATE plain SET value = 20 WHERE false',
	'DELETE FROM plain WHERE false')
{
	my ($rc, $out, $err) = $node->psql('postgres', qq[
		DO \$\$
		DECLARE
			chunk_scans bigint;
			compression_scans bigint;
			extra_chunk_scans bigint;
			extra_compression_scans bigint;
		BEGIN
			EXECUTE '$query';
			chunk_scans := pg_stat_get_xact_numscans('_timescaledb_catalog.chunk_relid_key'::regclass);
			compression_scans := pg_stat_get_xact_numscans('_timescaledb_catalog.compression_settings_compress_relid_idx'::regclass);
			FOR i IN 1..20 LOOP
				EXECUTE '$query';
			END LOOP;
			extra_chunk_scans := pg_stat_get_xact_numscans('_timescaledb_catalog.chunk_relid_key'::regclass) - chunk_scans;
			extra_compression_scans := pg_stat_get_xact_numscans('_timescaledb_catalog.compression_settings_compress_relid_idx'::regclass) - compression_scans;
			IF extra_chunk_scans <> 0 OR extra_compression_scans <> 0
			THEN
				RAISE EXCEPTION 'repeated lookups for a warm ordinary table: % chunk, % compression', extra_chunk_scans, extra_compression_scans;
			END IF;
		END
		\$\$;
	]);
	is($rc, 0, "no repeated metadata scans: $query") or diag($err);
}

# Keep the negative classification in another backend, then turn the same OID
# into a chunk. A direct INSERT must still get ModifyHypertable after attachment.
my $observer = $node->background_psql('postgres');
$observer->query_safe('SET client_min_messages = warning');
$observer->query_safe('SELECT * FROM plain; INSERT INTO plain SELECT 2, 20 WHERE false');
$node->safe_psql('postgres', q[
	CALL attach_chunk('ht', 'plain', '{"time": [0, 10]}');
]);
like($observer->query_safe('EXPLAIN (COSTS OFF) INSERT INTO plain VALUES (2, 20)'),
	qr/Custom Scan \(ModifyHypertable\)/,
	'cross-backend attachment invalidates a cached ordinary relation');
is($observer->query_safe('SELECT sum(value) FROM ht'), '10',
	'attached data is visible through the hypertable');
$node->safe_psql('postgres', "CALL detach_chunk('plain')");
unlike($observer->query_safe('EXPLAIN (COSTS OFF) INSERT INTO plain VALUES (2, 20)'),
	qr/ModifyHypertable/, 'detached relation is ordinary again');

# A same-backend conversion and rollback must not leave either classification
# behind, including when the OID survives a savepoint rollback.
$observer->query_safe('BEGIN; SAVEPOINT s');
$observer->query_safe("SELECT create_hypertable('plain', 'time', migrate_data => true)");
like($observer->query_safe('EXPLAIN (COSTS OFF) INSERT INTO plain VALUES (2, 20)'),
	qr/Custom Scan \(ModifyHypertable\)/, 'conversion invalidates the negative entry');
$observer->query_safe('ROLLBACK TO s');
unlike($observer->query_safe('EXPLAIN (COSTS OFF) INSERT INTO plain VALUES (2, 20)'),
	qr/ModifyHypertable/, 'savepoint rollback restores ordinary classification');
$observer->query_safe('ROLLBACK');

# An existing ordinary table can become compressed storage without changing
# its OID. A cached negative lookup must not bypass frozen-chunk protection.
$node->safe_psql('postgres', q[
	CREATE TABLE compressed_ht(time bigint NOT NULL, device text, value integer);
	SELECT create_hypertable('compressed_ht', 'time', chunk_time_interval => 10);
	ALTER TABLE compressed_ht SET (timescaledb.compress,
		timescaledb.compress_segmentby = 'device', timescaledb.compress_orderby = 'time');
	INSERT INTO compressed_ht VALUES (1, 'd1', 10);
	DO $setup$
	DECLARE
		chunk_rel regclass;
		compressed_rel regclass;
	BEGIN
		SELECT show_chunks('compressed_ht') INTO STRICT chunk_rel;
		PERFORM compress_chunk(chunk_rel);
		SELECT compress_relid::regclass INTO STRICT compressed_rel
		FROM _timescaledb_catalog.compression_settings WHERE relid = chunk_rel;
		EXECUTE format('CREATE TABLE compressed_plain (LIKE %s)', compressed_rel);
		EXECUTE format('INSERT INTO compressed_plain SELECT * FROM %s', compressed_rel);
		PERFORM decompress_chunk(chunk_rel);
	END
	$setup$;
]);
$observer->query_safe('EXPLAIN (COSTS OFF) INSERT INTO compressed_plain DEFAULT VALUES');
$node->safe_psql('postgres', q[
	SELECT _timescaledb_functions.create_compressed_chunk(
		ch, 'compressed_plain'::regclass,
		8192, 8192, 16384, 8192, 8192, 16384, 1, 1)
	FROM show_chunks('compressed_ht') AS ch;
	SELECT _timescaledb_functions.freeze_chunk(ch)
	FROM show_chunks('compressed_ht') AS ch;
]);
is($observer->query_safe(q[
	DO $check$
	BEGIN
		BEGIN
			EXECUTE 'EXPLAIN (COSTS OFF) INSERT INTO compressed_plain DEFAULT VALUES';
		EXCEPTION WHEN object_not_in_prerequisite_state THEN
			IF SQLERRM <> 'cannot modify compressed chunk belonging to a frozen chunk' THEN
				RAISE;
			END IF;
			RETURN;
		END;
		RAISE EXCEPTION 'cached ordinary classification bypassed frozen chunk protection';
	END
	$check$;
	SELECT 'blocked';
]), 'blocked', 'compressed attachment preserves frozen-chunk protection');
$observer->quit;

$node->safe_psql('postgres', q[
	CREATE TABLE chunk_pin(time bigint NOT NULL, value integer);
	SELECT create_hypertable('chunk_pin', 'time', chunk_time_interval => 1);
	INSERT INTO chunk_pin SELECT i, i FROM generate_series(0, 19) AS i;
	INSERT INTO chunk_pin SELECT i, i FROM generate_series(0, 19) AS i;
]);
is($node->safe_psql('postgres', 'SELECT count(*) FROM chunk_pin'), '40',
	'chunk creation under a live cache pin preserves every row');
is($node->safe_psql('postgres', "SELECT count(*) FROM show_chunks('chunk_pin')"),
	'20', 'chunk cache invalidation does not duplicate chunks');

done_testing();
