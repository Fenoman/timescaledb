# This file and its contents are licensed under the Timescale License.
# Please see the included NOTICE for copyright information and
# LICENSE-TIMESCALE for a copy of the license.

use strict;
use warnings;
use TimescaleNode;
use Test::More;

plan skip_all => 'set CITUS_COLUMNAR_LIBDIR to run the Citus compatibility test'
  unless $ENV{CITUS_COLUMNAR_LIBDIR};
plan skip_all => 'Citus columnar compatibility is tested only with PostgreSQL 16'
  unless ($ENV{PG_VERSION_MAJOR} // '') eq '16';

plan tests => 4;

my $node = TimescaleNode->new('citus_columnar_customscan');
$node->init;
$node->append_conf(
	'postgresql.conf',
	"dynamic_library_path='$ENV{CITUS_COLUMNAR_LIBDIR}:\$libdir'"
);
$node->append_conf('postgresql.conf',
	"shared_preload_libraries='citus_columnar,timescaledb'");
$node->append_conf('postgresql.conf',
	"max_parallel_workers=8\nmax_parallel_workers_per_gather=4");
$node->start;
$node->safe_psql('postgres', 'CREATE EXTENSION timescaledb');

$node->safe_psql(
	'postgres', q[
	CREATE TABLE metrics(time timestamptz NOT NULL, value bigint NOT NULL);
	SELECT create_hypertable('metrics', 'time', chunk_time_interval => INTERVAL '1 day');
	ALTER TABLE metrics SET (
		timescaledb.compress,
		timescaledb.compress_orderby = 'time');
	INSERT INTO metrics
	SELECT '2020-01-01 00:00:00+00'::timestamptz
			 + day * INTERVAL '1 day'
			 + value * INTERVAL '1 second',
		   value
	FROM generate_series(0, 15) AS day,
		 generate_series(1, 5000) AS value;
	SELECT count(compress_chunk(chunk))
	FROM show_chunks('metrics') AS chunk;
]);

my $serial_result = $node->safe_psql(
	'postgres', q[
	SET max_parallel_workers_per_gather = 0;
	SELECT count(*) || '|' || sum(value) FROM metrics;
]);

my $parallel_sql = q[
	SET debug_parallel_query = on;
	SET min_parallel_table_scan_size = 0;
	SET parallel_setup_cost = 0;
	SET parallel_tuple_cost = 0;
	SET parallel_leader_participation = off;
	SELECT count(*) || '|' || sum(value) FROM metrics;
];

my $parallel_result = $node->safe_psql('postgres', $parallel_sql);
is($parallel_result, '80000|200040000',
	'parallel aggregate returns the complete result');
is($parallel_result, $serial_result, 'parallel and serial aggregates match');

my $explain = $node->safe_psql(
	'postgres', qq[
	SET debug_parallel_query = on;
	SET min_parallel_table_scan_size = 0;
	SET parallel_setup_cost = 0;
	SET parallel_tuple_cost = 0;
	SET parallel_leader_participation = off;
	EXPLAIN (ANALYZE, VERBOSE, COSTS OFF, SUMMARY OFF, TIMING OFF)
	SELECT count(*) || '|' || sum(value) FROM metrics;
]);

like($explain, qr/Custom Scan \(TimescaleDBColumnarScan\)/,
	'explain uses the Timescale columnar scan');
like($explain, qr/Workers Launched: [1-9]/,
	'explain shows launched parallel workers');

done_testing();

1;
