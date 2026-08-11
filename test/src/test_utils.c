/*
 * This file and its contents are licensed under the Apache License 2.0.
 * Please see the included NOTICE for copyright information and
 * LICENSE-APACHE for a copy of the license.
 */
#include "test_utils.h"

#include <postgres.h>

#include <compat/compat.h>
#include <commands/explain.h>
#include <commands/dbcommands.h>
#include <executor/executor.h>
#if PG19_GE
#include <catalog/pg_database.h>
#endif
#include <fmgr.h>
#include <miscadmin.h>
#include <storage/latch.h>
#include <storage/proc.h>
#include <storage/procarray.h>
#include <utils/builtins.h>
#include <utils/elog.h>
#include <utils/guc.h>
#include <utils/memutils.h>

#include "debug_point.h"
#include "extension_constants.h"
#include "nodes/modify_hypertable.h"
#include "utils.h"

TS_FUNCTION_INFO_V1(ts_test_error_injection);
TS_FUNCTION_INFO_V1(ts_debug_shippable_error_after_n_rows);
TS_FUNCTION_INFO_V1(ts_debug_shippable_fatal_after_n_rows);

/*
 * Reproduce extensions which inspect a plan immediately after ExecutorStart
 * has initialized it. In particular, ModifyHypertable defers creation of its
 * ChunkTupleRouting state until the first executor call, so an early EXPLAIN
 * must tolerate a NULL state->ctr.
 *
 * The analyze flavor reproduces extensions which set analyze in their own
 * ExplainState regardless of whether the executor was started with
 * instrumentation, so explain callbacks must tolerate a NULL instrument.
 */
static ExecutorStart_hook_type previous_executor_start_hook = NULL;
static bool explain_in_executor_start_enabled = false;
static bool explain_analyze_in_executor_start_enabled = false;

static void
test_explain_in_executor_start(QueryDesc *query_desc, int eflags)
{
	if (previous_executor_start_hook)
	{
		previous_executor_start_hook(query_desc, eflags);
	}
	else
	{
		standard_ExecutorStart(query_desc, eflags);
	}

	if (explain_in_executor_start_enabled &&
		(query_desc->operation == CMD_INSERT || query_desc->operation == CMD_MERGE))
	{
		ExplainState *es = NewExplainState();
		ExplainPrintPlan(es, query_desc);
	}

	if (explain_analyze_in_executor_start_enabled && query_desc->operation == CMD_SELECT)
	{
		ExplainState *es = NewExplainState();
		es->analyze = true;
		es->verbose = true;
		ExplainPrintPlan(es, query_desc);
	}
}

static void
update_explain_in_executor_start_hook(void)
{
	bool enable =
		explain_in_executor_start_enabled || explain_analyze_in_executor_start_enabled;
	bool installed = ExecutorStart_hook == test_explain_in_executor_start;

	if (enable && !installed)
	{
		previous_executor_start_hook = ExecutorStart_hook;
		ExecutorStart_hook = test_explain_in_executor_start;
	}
	else if (!enable && installed)
	{
		ExecutorStart_hook = previous_executor_start_hook;
		previous_executor_start_hook = NULL;
	}
}

TS_TEST_FN(ts_test_enable_explain_in_executor_start)
{
	explain_in_executor_start_enabled = true;
	update_explain_in_executor_start_hook();

	PG_RETURN_VOID();
}

TS_TEST_FN(ts_test_disable_explain_in_executor_start)
{
	explain_in_executor_start_enabled = false;
	update_explain_in_executor_start_hook();

	PG_RETURN_VOID();
}

TS_TEST_FN(ts_test_enable_explain_analyze_in_executor_start)
{
	explain_analyze_in_executor_start_enabled = true;
	update_explain_in_executor_start_hook();

	PG_RETURN_VOID();
}

TS_TEST_FN(ts_test_disable_explain_analyze_in_executor_start)
{
	explain_analyze_in_executor_start_enabled = false;
	update_explain_in_executor_start_hook();

	PG_RETURN_VOID();
}

/*
 * Reproduce extensions which inspect an executed plan more than once before
 * ExecutorEnd. ModifyHypertable's EXPLAIN callback must leave both its saved
 * targetlists and its accumulated counters unchanged by the second poll.
 */
static ExecutorRun_hook_type previous_executor_run_hook = NULL;
static bool explain_in_executor_run_enabled = false;

static void
test_explain_in_executor_run(QueryDesc *query_desc, ScanDirection direction, uint64 count
#if PG18_LT
							 , bool execute_once
#endif
)
{
	if (previous_executor_run_hook)
	{
		previous_executor_run_hook(query_desc, direction, count
#if PG18_LT
							   , execute_once
#endif
		);
	}
	else
	{
		standard_ExecutorRun(query_desc, direction, count
#if PG18_LT
							 , execute_once
#endif
		);
	}

	if (explain_in_executor_run_enabled &&
		(query_desc->operation == CMD_DELETE || query_desc->operation == CMD_INSERT))
	{
		PlanState *planstate = query_desc->planstate;
		if (!ts_is_modify_hypertable_plan(planstate->plan))
		{
			elog(ERROR, "expected a ModifyHypertable plan");
		}

		ModifyHypertableState *state = (ModifyHypertableState *) planstate;
		if (query_desc->operation == CMD_INSERT)
		{
			Assert(state->ctr != NULL && state->ctr->counters != NULL);
			state->ctr->counters->batches_scanned = 1;
		}

		ExplainState *es = NewExplainState();
		es->verbose = true;
		ExplainPrintPlan(es, query_desc);
		List *saved_tlist = state->explain_saved_tlist;
		List *saved_custom_scan_tlist = state->explain_saved_custom_scan_tlist;
		if (query_desc->operation == CMD_DELETE && saved_tlist == NULL)
		{
			elog(ERROR, "runtime EXPLAIN did not save a ModifyHypertable targetlist");
		}
		ExplainPrintPlan(es, query_desc);

		if (query_desc->operation == CMD_DELETE &&
			(state->explain_saved_tlist != saved_tlist ||
			 (saved_custom_scan_tlist &&
			  state->explain_saved_custom_scan_tlist != saved_custom_scan_tlist)))
		{
			elog(ERROR, "repeated runtime EXPLAIN lost the ModifyHypertable targetlist");
		}
		if (query_desc->operation == CMD_INSERT && state->batches_scanned != 1)
		{
			elog(ERROR,
				 "repeated runtime EXPLAIN double-counted counters: expected 1, got %lld",
				 (long long) state->batches_scanned);
		}
	}
}

TS_TEST_FN(ts_test_enable_explain_in_executor_run)
{
	if (!explain_in_executor_run_enabled)
	{
		previous_executor_run_hook = ExecutorRun_hook;
		ExecutorRun_hook = test_explain_in_executor_run;
		explain_in_executor_run_enabled = true;
	}

	PG_RETURN_VOID();
}

TS_TEST_FN(ts_test_disable_explain_in_executor_run)
{
	if (explain_in_executor_run_enabled)
	{
		Assert(ExecutorRun_hook == test_explain_in_executor_run);
		ExecutorRun_hook = previous_executor_run_hook;
		previous_executor_run_hook = NULL;
		explain_in_executor_run_enabled = false;
	}

	PG_RETURN_VOID();
}

/*
 * Test assertion macros.
 *
 * Errors are expected since we want to test that the macros work. For each
 * macro, test one failing and one non-failing condition. The non-failing must
 * come first since the failing one will abort the function.
 */
TS_TEST_FN(ts_test_utils_condition)
{
	bool true_value = true;
	bool false_value = false;

	TestAssertTrue(true_value == true_value);
	TestAssertTrue(true_value == false_value);

	PG_RETURN_VOID();
}

TS_TEST_FN(ts_test_utils_int64_eq)
{
	int64 big = 32532978;
	int64 small = 3242234;

	TestAssertInt64Eq(big, small);
	TestAssertInt64Eq(big, big);

	PG_RETURN_VOID();
}

TS_TEST_FN(ts_test_utils_ptr_eq)
{
	bool true_value = true;
	bool false_value = false;
	bool *true_ptr = &true_value;
	bool *false_ptr = &false_value;

	TestAssertPtrEq(true_ptr, true_ptr);
	TestAssertPtrEq(true_ptr, false_ptr);

	PG_RETURN_VOID();
}

TS_TEST_FN(ts_test_utils_double_eq)
{
	double big_double = 923423478.3242;
	double small_double = 324.3;

	TestAssertDoubleEq(big_double, big_double);
	TestAssertDoubleEq(big_double, small_double);

	PG_RETURN_VOID();
}

Datum
ts_test_error_injection(PG_FUNCTION_ARGS)
{
	text *name = PG_GETARG_TEXT_PP(0);
	DEBUG_ERROR_INJECTION(text_to_cstring(name));
	PG_RETURN_VOID();
}

static int
transaction_row_counter(void)
{
	static LocalTransactionId last_lxid = 0;
	static int rows_seen = 0;
#if PG17_GE
	if (last_lxid != MyProc->vxid.lxid)
	{
		/* Reset it for each new transaction for predictable results. */
		rows_seen = 0;
		last_lxid = MyProc->vxid.lxid;
	}
#else
	if (last_lxid != MyProc->lxid)
	{
		rows_seen = 0;
		last_lxid = MyProc->lxid;
	}
#endif

	return rows_seen++;
}

static int
throw_after_n_rows(int max_rows, int severity)
{
	int rows_seen = transaction_row_counter();

	if (max_rows <= rows_seen)
	{
		ereport(severity,
				(errmsg("debug point: requested to error out after %d rows, %d rows seen",
						max_rows,
						rows_seen)));
	}

	return rows_seen;
}

Datum
ts_debug_shippable_error_after_n_rows(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT32(throw_after_n_rows(PG_GETARG_INT32(0), ERROR));
}

Datum
ts_debug_shippable_fatal_after_n_rows(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT32(throw_after_n_rows(PG_GETARG_INT32(0), FATAL));
}

/*
 * After how many rows should we error out according to the user-set option.
 */
static int
get_error_after_rows()
{
	int error_after = 7103; /* default is an arbitrary prime */

	const char *error_after_option =
		GetConfigOption(MAKE_EXTOPTION("debug_broken_sendrecv_error_after"), true, false);
	if (error_after_option)
	{
		error_after = pg_strtoint32(error_after_option);
	}

	return error_after;
}

/*
 * Broken send/receive functions for int4 that throw after an (arbitrarily
 * chosen prime or configured) number of rows.
 */
static void
broken_sendrecv_throw()
{
	/*
	 * Use ERROR, not FATAL, because PG versions < 14 are unable to report a
	 * FATAL error to the access node before closing the connection, so the test
	 * results would be different.
	 */
	(void) throw_after_n_rows(get_error_after_rows(), ERROR);
}

TS_FUNCTION_INFO_V1(ts_debug_broken_int4recv);

Datum
ts_debug_broken_int4recv(PG_FUNCTION_ARGS)
{
	broken_sendrecv_throw();
	return int4recv(fcinfo);
}

TS_FUNCTION_INFO_V1(ts_debug_broken_int4send);

Datum
ts_debug_broken_int4send(PG_FUNCTION_ARGS)
{
	broken_sendrecv_throw();
	return int4send(fcinfo);
}

/* An incorrect int4out that sometimes returns not a number. */
TS_FUNCTION_INFO_V1(ts_debug_incorrect_int4out);

Datum
ts_debug_incorrect_int4out(PG_FUNCTION_ARGS)
{
	int rows_seen = transaction_row_counter();

	if (rows_seen >= get_error_after_rows())
	{
		PG_RETURN_CSTRING("surprise");
	}

	return int4out(fcinfo);
}

/* Sleeps after a certain number of calls. */
static void
ts_debug_sleepy_function()
{
	static LocalTransactionId last_lxid = 0;
	static int rows_seen = 0;

#if PG17_GE
	if (last_lxid != MyProc->vxid.lxid)
	{
		/* Reset it for each new transaction for predictable results. */
		rows_seen = 0;
		last_lxid = MyProc->vxid.lxid;
	}
#else
	if (last_lxid != MyProc->lxid)
	{
		rows_seen = 0;
		last_lxid = MyProc->lxid;
	}
#endif

	rows_seen++;

	if (rows_seen >= 997)
	{
		(void) WaitLatch(MyLatch,
						 WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
						 1000,
						 /* wait_event_info = */ 0);
		ResetLatch(MyLatch);

		rows_seen = 0;
	}
}

TS_FUNCTION_INFO_V1(ts_debug_sleepy_int4recv);

Datum
ts_debug_sleepy_int4recv(PG_FUNCTION_ARGS)
{
	ts_debug_sleepy_function();
	return int4recv(fcinfo);
}

TS_FUNCTION_INFO_V1(ts_debug_sleepy_int4send);

Datum
ts_debug_sleepy_int4send(PG_FUNCTION_ARGS)
{
	ts_debug_sleepy_function();
	return int4send(fcinfo);
}

TS_FUNCTION_INFO_V1(ts_bgw_wait);
Datum
ts_bgw_wait(PG_FUNCTION_ARGS)
{
	text *datname = PG_GETARG_TEXT_PP(0);
	/* The timeout is given in seconds, so we compute the number of iterations
	 * necessary to get a coverage of that time */
	uint32 iterations = PG_ARGISNULL(1) ? 5 : (PG_GETARG_UINT32(1) + 4) / 5;
	bool raise_error = PG_ARGISNULL(2) ? true : PG_GETARG_BOOL(2);
	Oid dboid = get_database_oid(text_to_cstring(datname), false);

	/* This function contains a timeout of 5 seconds, so we iterate a few
	 * times to make sure that it really has terminated. */
	int notherbackends = 0;
	int npreparedxacts = 0;
	while (iterations-- > 0)
	{
		if (!CountOtherDBBackends(dboid, &notherbackends, &npreparedxacts))
		{
			PG_RETURN_NULL();
		}
		ereport(NOTICE,
				(errmsg("source database \"%s\" is being accessed by other users",
						text_to_cstring(datname)),
				 errdetail("There are %d other session(s) and %d prepared transaction(s) using the "
						   "database.",
						   notherbackends,
						   npreparedxacts)));
	}

	if (raise_error)
	{
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_IN_USE),
				 errmsg("source database \"%s\" is being accessed by other users",
						text_to_cstring(datname)),
				 errdetail("There are %d other session(s) and %d prepared transaction(s) using the "
						   "database.",
						   notherbackends,
						   npreparedxacts)));
	}

	pg_unreachable();
}

/*
 * Return the number of bytes allocated in a given memory context and its
 * children.
 */
TS_FUNCTION_INFO_V1(ts_debug_allocated_bytes);
Datum
ts_debug_allocated_bytes(PG_FUNCTION_ARGS)
{
	MemoryContext context = NULL;
	char *context_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
	if (strcmp(context_name, "PortalContext") == 0)
	{
		context = PortalContext;
	}
	else if (strcmp(context_name, "CacheMemoryContext") == 0)
	{
		context = CacheMemoryContext;
	}
	else if (strcmp(context_name, "TopMemoryContext") == 0)
	{
		context = TopMemoryContext;
	}
	else
	{
		ereport(ERROR,
				(errmsg("unknown memory context '%s' (search for arbitrary contexts by name is not"
						"implemented)",
						context_name)));
		PG_RETURN_NULL();
	}

	PG_RETURN_UINT64(MemoryContextMemAllocated(context, /* recurse = */ true));
}

TS_TEST_FN(ts_test_errdata_to_jsonb)
{
	ErrorData *edata = (ErrorData *) palloc(sizeof(ErrorData));
	edata->elevel = ERROR;
	edata->output_to_server = true;
	edata->output_to_client = true;
	edata->hide_stmt = false;
	edata->hide_ctx = false;
	edata->filename = "test error filename";
	edata->lineno = 123;
	edata->funcname = "test error function";
	edata->domain = "test error domain";
	edata->context_domain = "test error context domain";
	edata->sqlerrcode = ERRCODE_INVALID_PARAMETER_VALUE;
	edata->message = "test error message";
	edata->detail = "test error detail";
	edata->detail_log = "test error detail log";
	edata->hint = "test error hint";
	edata->context = "test error context";
	edata->backtrace = "test error backtrace";
	edata->message_id = "test error message id";
	edata->schema_name = "test error schema";
	edata->table_name = "test error table";
	edata->column_name = "test error column";
	edata->datatype_name = "test error datatype";
	edata->constraint_name = "test error constraint";
	edata->cursorpos = 42;
	edata->internalpos = 42;
	edata->internalquery = "test error internal query";
	edata->saved_errno = 42;

	NameData proc_schema = { .data = { 0 } };
	NameData proc_name = { .data = { 0 } };
	namestrcpy(&proc_schema, "proc_schema");
	namestrcpy(&proc_name, "proc_name");

	Jsonb *out = ts_errdata_to_jsonb(edata, &proc_schema, &proc_name);

	PG_RETURN_JSONB_P(out);
}
