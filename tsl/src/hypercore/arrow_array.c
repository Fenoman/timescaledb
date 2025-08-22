/*
 * This file and its contents are licensed under the Timescale License.
 * Please see the included NOTICE for copyright information and
 * LICENSE-TIMESCALE for a copy of the license.
 */

#include <postgres.h>
#include <access/tupmacs.h>
#include <fmgr.h>
#include <utils/datum.h>
#include <utils/palloc.h>

#include "arrow_array.h"
#include "compression/arrow_c_data_interface.h"
#include "compression/compression.h"
#include "src/utils.h"

#define TYPLEN_VARLEN (-1)
#define ARROW_VARLEN_MAX_PAYLOAD (64 * 1024 * 1024) /* 64MB reasonable limit for varlena payload */

typedef struct ArrowPrivate
{
	MemoryContext mcxt; /* The memory context on which the private data is allocated */
	size_t value_capacity;
	struct varlena *value; /* For text types, a reusable memory area to create
							* the varlena version of the c-string */
	bool typbyval;		   /* Cached typbyval for the type in the arrow array. This
							* avoids having to do get_typbyval() syscache lookups on
							* hot paths. */
	int8 text_format_hint; /* TEXT format cache: -1=unknown, 0=cstring (new), 1=varlena (legacy).
							* Determined on first non-null TEXT value to avoid repeated format
							* detection for subsequent values. */
} ArrowPrivate;

static Datum
arrow_private_cstring_to_text_datum(ArrowPrivate *ap, const uint8 *data, size_t datalen)
{
	const size_t varlen = VARHDRSZ + datalen;

	/* Allocate memory on the ArrowArray's memory context. Start with twice
	 * the size necessary for the value. Reallocate and expand later as
	 * necessary for next values. */
	if (ap->value == NULL)
	{
		ap->value_capacity = varlen * 2;
		ap->value = MemoryContextAlloc(ap->mcxt, ap->value_capacity);
	}
	else if (varlen > ap->value_capacity)
	{
		ap->value_capacity = varlen * 2;
		ap->value = repalloc(ap->value, ap->value_capacity);
	}

	SET_VARSIZE(ap->value, varlen);
	memcpy(VARDATA_ANY(ap->value), data, datalen);

	return PointerGetDatum(ap->value);
}

static ArrowPrivate *
arrow_private_create(ArrowArray *array, Oid typid)
{
	ArrowPrivate *private;

	Assert(NULL == array->private_data);
	private = palloc0(sizeof(ArrowPrivate));
	private->mcxt = CurrentMemoryContext;
	private->text_format_hint = -1; /* Unknown format initially */
	
	/* Set private_data BEFORE potentially throwing operation to ensure
	 * proper cleanup in case of exception */
	array->private_data = private;
	
	/* This call might throw an exception if typid is invalid */
	private->typbyval = get_typbyval(typid);

	return private;
}

static inline ArrowPrivate *
arrow_private_get(const ArrowArray *array)
{
	Assert(array->private_data != NULL);
	return (ArrowPrivate *) array->private_data;
}

static void
arrow_private_release(ArrowArray *array)
{
	if (array->private_data != NULL)
	{
		ArrowPrivate *ap = array->private_data;

		if (ap->value != NULL)
		{
			pfree(ap->value);
			ap->value = NULL;
		}
		pfree(ap);
		array->private_data = NULL;
	}
}

/*
 * Extend a buffer if necessary.
 *
 * We double the memory because we want to amortize the allocation cost to so
 * that it becomes O(n). The new memory will be allocated in the same memory
 * context as the memory was originally allocated in.
 */
#define EXTEND_BUFFER_IF_NEEDED(BUFFER, NEEDED, CAPACITY)                                          \
	do                                                                                             \
	{                                                                                              \
		if ((unsigned long) (NEEDED) >= (unsigned long) (CAPACITY))                                \
		{                                                                                          \
			/* First check if NEEDED itself is too large */                                       \
			if ((unsigned long) (NEEDED) > INT64_MAX)                                             \
				ereport(ERROR,                                                                     \
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),                                 \
						 errmsg("requested buffer size too large: %lu", (unsigned long)(NEEDED))));\
			                                                                                       \
			/* Calculate new capacity safely */                                                   \
			size_t new_capacity = (CAPACITY);                                                     \
			while (new_capacity <= (unsigned long) (NEEDED))                                      \
			{                                                                                      \
				/* Check for overflow BEFORE doubling */                                          \
				if (new_capacity > SIZE_MAX / 2 || new_capacity > INT64_MAX / 2)                  \
					ereport(ERROR,                                                                 \
							(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),                             \
							 errmsg("arrow buffer size overflow: cannot double %zu",              \
									new_capacity)));                                               \
				new_capacity *= 2;                                                                 \
			}                                                                                      \
			(CAPACITY) = new_capacity;                                                             \
			(BUFFER) = repalloc((BUFFER), (CAPACITY));                                             \
		}                                                                                          \
	} while (0)

/*
 * Release buffer memory.
 */
static void
arrow_release_buffers(ArrowArray *array)
{
	/*
	 * The recommended release function frees child nodes and the dictionary
	 * in the Arrow array, but, currently, the child array is not used so we
	 * do not care about it.
	 */
	Assert(array->children == NULL);

	for (int64 i = 0; i < array->n_buffers; ++i)
	{
		/* Validity bitmap might be NULL even if it is counted
		 * in n_buffers, so need to check for NULL values. */
		if (array->buffers[i] != NULL)
		{
			pfree((void *) array->buffers[i]);
			array->buffers[i] = NULL; /* Just a precaution to avoid a dangling reference */
		}
	}

	array->n_buffers = 0;

	if (array->dictionary)
	{
		arrow_release_buffers(array->dictionary);
		array->dictionary = NULL;
	}

	if (array->private_data)
		arrow_private_release(array);
}

/*
 * Variable-size primitive layout ArrowArray from decompression iterator.
 */
static ArrowArray *
arrow_from_iterator_varlen(MemoryContext mcxt, DecompressionIterator *iterator, Oid typid)
{
	/* Starting capacity of the offset buffer in bytes. This is probably an
	 * over-estimation in some cases, but avoids reallocation for the common case. */
	int64 offsets_capacity = sizeof(int32) * (TARGET_COMPRESSED_BATCH_SIZE + 1);
	int64 data_capacity = 4 * offsets_capacity; /* Starting capacity of the data buffer in bytes */
    /*
     * Validity bitmap is addressed in 64-bit words (see arrow_set_row_validity).
     * Start with capacity for at least one 64-bit word and grow as needed.
     */
    int64 validity_capacity = sizeof(uint64) * 8; /* Start with 8 words = 512 bits */
	int32 endpos = 0; /* Can be 32 or 64 bits signed integers */
	int64 array_length;
	int64 null_count = 0;
    int32 *offsets_buffer = MemoryContextAlloc(mcxt, offsets_capacity);
    uint8 *data_buffer = MemoryContextAlloc(mcxt, data_capacity);
    /* Zero to keep unused tail bits clean for popcount-based consumers */
    uint64 *validity_buffer = MemoryContextAllocZero(mcxt, validity_capacity);

	/* Just a precaution: type should be varlen */
	Assert(get_typlen(typid) == TYPLEN_VARLEN);

	/* First offset is always zero and there are length + 1 offsets */
	offsets_buffer[0] = 0;

	for (array_length = 0;; ++array_length)
	{
		DecompressResult result = iterator->try_next(iterator);

		if (result.is_done)
			break;

		TS_DEBUG_LOG("storing %s varlen value row " INT64_FORMAT
					 " at offset %d (varlen size %lu, offset "
					 "capacity " INT64_FORMAT ", data capacity " INT64_FORMAT ")",
					 datum_as_string(typid, result.val, result.is_null),
					 array_length,
					 endpos,
					 result.is_null ? 0 : (unsigned long) VARSIZE_ANY(result.val), /* cast for 32-bit builds */
					 offsets_capacity,
					 data_capacity);

		/* Check for potential overflow in array_length + 2 (used for offsets buffer) */
		if (array_length >= INT64_MAX - 2)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("array length overflow")));
        /*
         * Offsets buffer contains (array_length + 1) valid indices and we will
         * write index [array_length + 1] below, so ensure space for (array_length + 2)
         * elements in total.
         */
        EXTEND_BUFFER_IF_NEEDED(offsets_buffer,
                                sizeof(*offsets_buffer) * (array_length + 2),
                                offsets_capacity);
        /*
         * We are about to set bit for row number = array_length (0-based).
         * Ensure there is space for word index (array_length / 64).
         */
        EXTEND_BUFFER_IF_NEEDED(validity_buffer,
                                sizeof(uint64) * ((array_length / 64) + 1),
                                validity_capacity);

		arrow_set_row_validity(validity_buffer, array_length, !result.is_null);

		if (result.is_null)
			++null_count;
		else
		{
			/*
			 * For TEXT we store cstring payload (WITHOUT varlena header),
			 * as expected by arrow_get_datum_varlen(); for other varlena types
			 * we keep the current "with header" layout.
			 */
			const bool is_text = (typid == TEXTOID);
			const int payload_len = is_text ? VARSIZE_ANY_EXHDR(result.val)
											: VARSIZE_ANY(result.val);
			const char *src = is_text ? (const char *) VARDATA_ANY(result.val)
									  : (const char *) DatumGetPointer(result.val);

			/* Check for overflow BEFORE extending buffer to prevent buffer overflow vulnerability */
			if (payload_len < 0 || payload_len > PG_INT32_MAX)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("invalid varlen size: %d", payload_len)));

			/* Symmetric limit for TEXT payload - same as on read side for consistency */
			if (is_text && payload_len > ARROW_VARLEN_MAX_PAYLOAD)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("text payload %d exceeds reasonable limit", payload_len)));

			/* ensure payload fits into 32-bit offset space */
			if (endpos > PG_INT32_MAX - payload_len)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("arrow varlen buffer size overflow: endpos %d + len %d exceeds limit",
								endpos, payload_len)));

			EXTEND_BUFFER_IF_NEEDED(data_buffer, endpos + payload_len, data_capacity);
			memcpy(&data_buffer[endpos], src, payload_len);
			endpos += payload_len;
		}

		offsets_buffer[array_length + 1] = endpos;
	}

	ArrowArray *array = arrow_create_with_buffers(mcxt, 3);
	array->length = array_length;
	array->buffers[0] = validity_buffer;
	array->buffers[1] = offsets_buffer;
	array->buffers[2] = data_buffer;
	array->null_count = null_count;
	array->release = arrow_release_buffers;

	return array;
}

/*
 * Fixed-Size Primitive layout ArrowArray from decompression iterator.
 */
static ArrowArray *
arrow_from_iterator_fixlen(MemoryContext mcxt, DecompressionIterator *iterator, Oid typid,
						   int16 typlen)
{
	const bool typbyval = get_typbyval(typid);
	int64 data_capacity = 64 * typlen; /* Capacity of the data buffer */
    /* Validity bitmap grows in 64-bit words; Start with 8 words = 512 bits. */
    int64 validity_capacity = sizeof(uint64) * 8;
	uint8 *data_buffer = MemoryContextAlloc(mcxt, data_capacity * sizeof(uint8));
    /* Zero to keep unused tail bits clean for popcount-based consumers */
    uint64 *validity_buffer = MemoryContextAllocZero(mcxt, validity_capacity);
	int64 array_length;
	int64 null_count = 0;

	/* Just a precaution: this should not be a varlen type */
	Assert(typlen > 0);

	/*
	 * Bool type should never used iterator decompression because it has a
	 * bulk decompression implementation
	 */
	Assert(typid != BOOLOID);

	for (array_length = 0;; ++array_length)
	{
		DecompressResult result = iterator->try_next(iterator);

		if (result.is_done)
			break;

		/* Check for overflow before extending buffers */
		if (array_length >= INT64_MAX - 1)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("array length overflow")));
		
		/* Check for potential multiplication overflow */
		if (typlen > 0 && array_length >= (INT64_MAX / typlen) - 1)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("data buffer size overflow")));

        /* Ensure capacity for the 64-bit word containing bit array_length. */
        EXTEND_BUFFER_IF_NEEDED(validity_buffer,
                                sizeof(uint64) * ((array_length / 64) + 1),
                                validity_capacity);
		EXTEND_BUFFER_IF_NEEDED(data_buffer, typlen * (array_length + 1), data_capacity);

		arrow_set_row_validity(validity_buffer, array_length, !result.is_null);

		if (result.is_null)
			++null_count;
		else if (typbyval)
		{
			/*
			 * We use unsigned integers to avoid conversions between signed
			 * and unsigned values (which in theory could change the value)
			 * when converting to datum (which is an unsigned value).
			 *
			 * Conversions between unsigned values is well-defined in the C
			 * standard and will work here.
			 */
			switch (typlen)
			{
				case sizeof(uint8):
					data_buffer[array_length] = DatumGetUInt8(result.val);
					break;
				case sizeof(uint16):
					((uint16 *) data_buffer)[array_length] = DatumGetUInt16(result.val);
					break;
				case sizeof(uint32):
					((uint32 *) data_buffer)[array_length] = DatumGetUInt32(result.val);
					break;
				case sizeof(uint64):
					/* This branch is not called for by-reference 64-bit values */
					((uint64 *) data_buffer)[array_length] = DatumGetUInt64(result.val);
					break;
				default:
					ereport(ERROR,
							errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
							errmsg("not supporting writing by value length %d", typlen));
			}
		}
		else
		{
			memcpy(&data_buffer[typlen * array_length], DatumGetPointer(result.val), typlen);
		}
	}

	ArrowArray *array = arrow_create_with_buffers(mcxt, 2);
	array->length = array_length;
	array->buffers[0] = validity_buffer;
	array->buffers[1] = data_buffer;
	array->null_count = null_count;
	array->release = arrow_release_buffers;
	return array;
}

/*
 * Read the entire contents of a decompression iterator into the arrow array.
 */
static ArrowArray *
arrow_from_iterator(MemoryContext mcxt, DecompressionIterator *iterator, Oid typid, int16 typlen)
{
	if (typlen == TYPLEN_VARLEN)
		return arrow_from_iterator_varlen(mcxt, iterator, typid);
	else
		return arrow_from_iterator_fixlen(mcxt, iterator, typid, typlen);
}

static ArrowArray *
arrow_generic_decompress_all(Datum compressed, Oid typid, MemoryContext dest_mctx)
{
	const int16 typlen = get_typlen(typid);
	/* Slightly weird interface for passing the header, but this is what the
	 * other decompress_all functions are using. We might want to refactor
	 * this later. */
	const CompressedDataHeader *header =
		(const CompressedDataHeader *) PG_DETOAST_DATUM(compressed);
	DecompressionInitializer initializer =
		tsl_get_decompression_iterator_init(header->compression_algorithm, false);
	DecompressionIterator *iterator = initializer(compressed, typid);
	return arrow_from_iterator(dest_mctx, iterator, typid, typlen);
}

static DecompressAllFunction
arrow_get_decompress_all(uint8 compression_alg, Oid typid)
{
	DecompressAllFunction decompress_all = NULL;

	decompress_all = tsl_get_decompress_all_function(compression_alg, typid);

	if (decompress_all == NULL)
		decompress_all = arrow_generic_decompress_all;

	Assert(decompress_all != NULL);
	return decompress_all;
}

#ifdef USE_ASSERT_CHECKING
static bool
verify_offsets(const ArrowArray *array)
{
	/* Check offsets in the main array */
	if (array->n_buffers == 3 && array->length > 0)
	{
		const int32 *offsets = array->buffers[1];
		
		/* Validate that we have access to offsets buffer */
		if (offsets == NULL)
			return false;

		/* We need array->length + 1 entries in offsets buffer */
		/* Check each pair (offsets[i], offsets[i+1]) where i < array->length */
		for (int64 i = 0; i < array->length; ++i)
		{
			/* Safe to access offsets[i] and offsets[i+1] since i < array->length
			 * and offsets buffer should have array->length + 1 entries */
			if (offsets[i + 1] < offsets[i])
				return false;
		}
	}
	
	/* Also check dictionary offsets if dictionary exists */
	if (array->dictionary && array->dictionary->n_buffers == 3 && array->dictionary->length > 0)
	{
		const int32 *dict_offsets = array->dictionary->buffers[1];
		
		/* Validate that dictionary has offsets buffer */
		if (dict_offsets == NULL)
			return false;
		
		/* Check monotonicity of dictionary offsets */
		for (int64 i = 0; i < array->dictionary->length; ++i)
		{
			if (dict_offsets[i + 1] < dict_offsets[i])
				return false;
		}
	}
	
	return true;
}
#endif

ArrowArray *
arrow_from_compressed(Datum compressed, Oid typid, MemoryContext dest_mcxt, MemoryContext tmp_mcxt)
{
	/* Validate memory contexts to prevent confusion */
	if (dest_mcxt == NULL || tmp_mcxt == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid memory context for arrow decompression")));
	
	/*
	 * Memory Context Usage:
	 * - tmp_mcxt: For temporary allocations during decompression (will be reset)
	 * - dest_mcxt: For the final ArrowArray that will be returned
	 * - oldcxt: To restore the original context at the end
	 *
	 * Need to detoast on our temporary memory context because
	 * CurrentMemoryContext can be a per-tuple memory context which uses Bump
	 * allocator on PG17. The Bump allocator doesn't support pfree(), which is
	 * needed by detoasting since it does some catalog scans.
	 */
	MemoryContext oldcxt = MemoryContextSwitchTo(tmp_mcxt);
	const CompressedDataHeader *header = (CompressedDataHeader *) PG_DETOAST_DATUM(compressed);
	if (header->compression_algorithm == COMPRESSION_ALGORITHM_NULL)
	{
		/*
		 * The NULL compression algorithm represents all NULL values.
		 */
		MemoryContextSwitchTo(oldcxt);
		return NULL;
	}
	DecompressAllFunction decompress_all =
		arrow_get_decompress_all(header->compression_algorithm, typid);

	TS_DEBUG_LOG("decompressing column with type %s using decompression algorithm %s",
				 format_type_be(typid),
				 NameStr(*compression_get_algorithm_name(header->compression_algorithm)));

	ArrowArray *array = decompress_all(PointerGetDatum(header), typid, dest_mcxt);

	/* Validate array is not NULL BEFORE any access */
	if (array == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("decompression returned NULL array")));

	Assert(verify_offsets(array));

	/*
	 * If the release function is not set, it is the old-style decompress_all
	 * and then buffers should be deleted by default.
	 */
	if (array->release == NULL)
		array->release = arrow_release_buffers;
	
	MemoryContextSwitchTo(dest_mcxt);
	/* Create private arrow info on the same memory context as the array itself */
	arrow_private_create(array, typid);

	/*
	 * Reset temporary context to free any intermediate allocations.
	 * This is safe because all data we need to keep was allocated in dest_mcxt.
	 *
	 * The amount of data is bounded by the number of columns in the tuple
	 * table slot, so it might be possible to skip this reset.
	 */
	MemoryContextReset(tmp_mcxt);
	
	/* Restore the original memory context */
	MemoryContextSwitchTo(oldcxt);

	return array;
}

/*
 * Get varlen datum from arrow array.
 *
 * This will always be a reference.
 */
static NullableDatum
arrow_get_datum_varlen(const ArrowArray *array, Oid typid, uint16 index)
{
	/* Validate input parameters to prevent NULL pointer dereference */
	if (array == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("arrow array is NULL")));
	
	if (array->buffers == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("arrow array buffers are NULL")));
	
	/* Validate index bounds */
	if (index >= array->length)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("array index %u out of bounds, array length is %lld",
						index, (long long) array->length)));
	
	const uint64 *restrict validity = array->buffers[0];
	const int32 *offsets;
	const uint8 *data;
	Datum value;

	/* Check if validity bitmap is NULL - if so, all values are valid by Arrow spec */
	if (validity != NULL && !arrow_row_is_valid(validity, index))
		return (NullableDatum){ .isnull = true };

    if (array->dictionary)
    {
        const ArrowArray *dict = array->dictionary;
        
        /* Validate dictionary-indexed layout: index buffer present and layout sane */
        if (array->n_buffers < 2)
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("invalid arrow array layout: expected index buffer for dictionary")));

        /* Validate dictionary index buffer exists before access */
        if (array->buffers[1] == NULL)
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("dictionary index buffer is NULL")));
		
		/* 
		 * TimescaleDB uses int16 for dictionary indices to save memory.
		 * This limits dictionaries to 32767 unique values but is sufficient
		 * for time-series compression scenarios.
		 * TODO: Consider supporting int32 indices for external Arrow data import.
		 */
		const int16 *indexes = (int16 *) array->buffers[1];
		int16 dict_index = indexes[index];
		
		/* Validate dictionary index bounds */
		if (dict_index < 0 || dict_index >= dict->length)
			ereport(ERROR,
					(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
					 errmsg("dictionary index %d out of bounds, dictionary length is %lld",
							dict_index, (long long) dict->length)));
		
        /* Validate dictionary has required buffers (offsets + data) */
        if (dict->n_buffers < 3 || dict->buffers == NULL || dict->buffers[1] == NULL || dict->buffers[2] == NULL)
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("dictionary array buffers are NULL or incomplete")));
		
		index = dict_index;
		offsets = dict->buffers[1];
		data = dict->buffers[2];
	}
    else
    {
        /* Validate non-dictionary buffers exist */
        if (array->n_buffers < 3)
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("invalid arrow array layout: expected 3 buffers for varlen")));

        if (array->buffers[1] == NULL || array->buffers[2] == NULL)
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("arrow array buffers are incomplete (offsets/data)")));
		
		offsets = array->buffers[1];
		data = array->buffers[2];
	}

	const int32 offset = offsets[index];
	
	/* Get data buffer size from the last offset entry. 
	 * Use dictionary length if we're working with dictionary data */
	const int64 effective_length = array->dictionary ? array->dictionary->length : array->length;
	const int32 data_buffer_size = offsets[effective_length];
	
	/* Validate offset bounds */
	if (offset < 0 || offset > data_buffer_size)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("invalid offset %d, buffer size %d", offset, data_buffer_size)));

	/* TEXT is stored back-to-back without a varlena header in the new layout.
	 * However, legacy compressed blocks may still contain TEXT as a regular
	 * varlena (with 1B/4B header). Use cached format hint after first detection
	 * to avoid repeated format checks for subsequent values.
	 */
	if (typid == TEXTOID)
	{
	    ArrowPrivate *ap = arrow_private_get(array);

	    /* We will access offsets[index + 1], so ensure index < effective_length.
	     * effective_length is dictionary->length for dictionary-encoded arrays,
	     * otherwise it's array->length.
	     */
	    if (index >= effective_length)
	        ereport(ERROR,
	                (errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
	                 errmsg("cannot access next offset for index %u, array length %lld",
	                        index, (long long) effective_length)));

	    const int32 next_offset = offsets[index + 1];

	    /* Offsets must be monotonically non-decreasing and fit the data buffer.
	     * Validate both the current offset (checked earlier) and the next one.
	     */
	    if (next_offset < offset || next_offset > data_buffer_size)
	        ereport(ERROR,
	                (errcode(ERRCODE_DATA_CORRUPTED),
	                 errmsg("invalid next_offset %d for index %u", next_offset, index)));

	    const int32 datalen = next_offset - offset;
	    const uint8 *dp = &data[offset];
	    
	    /* Check cached format hint to avoid repeated detection */
	    if (ap->text_format_hint == -1)
	    {
	        /* Unknown format - need to detect on first non-empty value */
	        bool looks_varlena = false;

	        /* Heuristics: treat the slice as varlena only if the header is present
	         * and header+payload size matches exactly the slice length.
	         */
	        if (datalen >= (int32) VARHDRSZ_SHORT)
	        {
	            /* First byte determines if it's 1B or 4B header */
	            if (VARATT_IS_1B(dp))
	            {
	                /* 1-byte varlena header: size = payload only. Require exact match. */
	                const int32 exlen = VARSIZE_ANY_EXHDR(dp);
	                if (exlen >= 0 &&
	                    exlen <= ARROW_VARLEN_MAX_PAYLOAD &&
	                    exlen + (int32) VARHDRSZ_SHORT == datalen)
	                    looks_varlena = true;
	            }
	            else if (datalen >= (int32) VARHDRSZ)
	            {
	                /* 4-byte header: copy to aligned local variable to avoid
	                 * unaligned access on architectures with strict alignment */
	                union {
	                    struct varlena vl;
	                    char data[VARHDRSZ];
	                    varattrib_4b va4b;  /* Ensure proper size for VARSIZE_ANY macros */
	                } hdr;
	                memcpy(&hdr.data, dp, VARHDRSZ);
	                const char *hp = (const char *) &hdr.vl;
	                
	                /* Skip toasted/compressed/indirect headers */
	                if (!VARATT_IS_EXTERNAL(hp) &&
	                    !VARATT_IS_COMPRESSED(hp)
#ifdef VARATT_IS_INDIRECT
	                    && !VARATT_IS_INDIRECT(hp)
#endif
	                    )
	                {
	                    /* 4-byte varlena header: size includes header. Require exact match. */
	                    const int32 vlen  = VARSIZE_ANY(hp);
	                    const int32 exlen = VARSIZE_ANY_EXHDR(hp);
	                    if (vlen == datalen &&
	                        exlen <= ARROW_VARLEN_MAX_PAYLOAD)
	                        looks_varlena = true;
	                }
	            }
	        }
	        
	        /* Cache the detected format for subsequent values */
	        ap->text_format_hint = looks_varlena ? 1 : 0;
	    }
	    
	    /* Use cached format hint */
	    if (ap->text_format_hint == 1)
	    {
	        /* Legacy layout: buffer contains a full varlena (1B/4B header) */
	        const char *vp = (const char *) dp;
	        const int32 exlen = VARSIZE_ANY_EXHDR(vp);
	        value = arrow_private_cstring_to_text_datum(ap,
	                                                    (const uint8 *) VARDATA_ANY(vp),
	                                                    (size_t) exlen);
	    }
	    else
	    {
	        /* New layout: buffer contains a raw cstring payload (no varlena header) */
	        if (datalen > ARROW_VARLEN_MAX_PAYLOAD)
	            ereport(ERROR,
	                    (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
	                     errmsg("text payload %d exceeds reasonable limit", datalen)));

	        value = arrow_private_cstring_to_text_datum(ap, dp, (size_t) datalen);
	    }
	}
    else
    {
    	ArrowPrivate *ap = arrow_private_get(array); /* for memory context on aligned copy */

        /* For non-text types, validate that offset doesn't exceed buffer */
        if (offset >= data_buffer_size)
            ereport(ERROR,
                    (errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
                     errmsg("offset %d exceeds buffer bounds %d", offset, data_buffer_size)));

        /* Ensure we can read at least 1 header byte (short varlena case) */
        if (data_buffer_size - offset < (int32) VARHDRSZ_SHORT)
            ereport(ERROR,
                    (errcode(ERRCODE_DATA_CORRUPTED),
                     errmsg("insufficient bytes (%d) for varlena header at offset %d",
                            data_buffer_size - offset, offset)));

        /* Prepare pointer to raw data and check header type */
        const char *vp_raw = (const char *) &data[offset];
        bool is_1b = VARATT_IS_1B(vp_raw);
        
        /* If it's a 4B varlena, we also must have 4 bytes for the header */
        if (!is_1b && (data_buffer_size - offset < (int32) VARHDRSZ))
            ereport(ERROR,
                    (errcode(ERRCODE_DATA_CORRUPTED),
                     errmsg("insufficient bytes for 4-byte varlena header at offset %d", offset)));

        /* Read header via aligned local copy to avoid unaligned 4-byte loads
         * on architectures with strict alignment requirements */
        struct varlena hdrbuf;
        const char *vp = vp_raw;
        if (!is_1b)
        {
            /* Copy 4-byte header to aligned buffer for safe access */
            memcpy(&hdrbuf, vp_raw, VARHDRSZ);
            vp = (const char *) &hdrbuf;
        }
        
        /* Now safe to read sizes from aligned pointer */
        int32 vlen  = VARSIZE_ANY(vp);               /* size including header (1B or 4B) */
        int32 exlen = VARSIZE_ANY_EXHDR(vp);         /* payload size excluding header */

        const int32 hdrsz = is_1b ? (int32) VARHDRSZ_SHORT : (int32) VARHDRSZ;

        /* Sanity checks */
        if (vlen < hdrsz || vlen > (data_buffer_size - offset))
            ereport(ERROR,
                    (errcode(ERRCODE_DATA_CORRUPTED),
                     errmsg("varlena size %d out of bounds at offset %d (buffer %d)",
                            vlen, offset, data_buffer_size)));

        /* Reasonable payload limit */
        if (exlen > ARROW_VARLEN_MAX_PAYLOAD)
            ereport(ERROR,
                    (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                     errmsg("varlena payload %d exceeds reasonable limit", exlen)));

    	/*
         * If the varlena data pointer is not properly aligned, copy it into
         * MAXALIGN'ed memory. This is needed for the full varlena data,
         * not just the header, on strict-alignment architectures.
         * Note: We only need this for 4B headers since 1B headers can be
         * read byte-by-byte without alignment issues.
         */
        if (!is_1b && MAXIMUM_ALIGNOF > 1 && (((uintptr_t) vp_raw) & (MAXIMUM_ALIGNOF - 1)) != 0)
		{
			struct varlena *aligned = (struct varlena *) MemoryContextAlloc(ap->mcxt, vlen);
	        memcpy(aligned, vp_raw, vlen);
	        value = PointerGetDatum(aligned);
		}
    	else
			value = PointerGetDatum(vp_raw);
    }

	/* We have stored the bytes of the varlen value directly in the buffer, so
	 * this should work as expected. */
	TS_DEBUG_LOG("retrieved varlen value '%s' row %u"
				 " from offset %d dictionary=%p in memory context %s",
				 datum_as_string(typid, value, false),
				 index,
				 offset,
				 array->dictionary,
				 GetMemoryChunkContext((void *) data)->name);

	return (NullableDatum){ .isnull = false, .value = value };
}

/*
 * Get a fixed-length datum from the arrow array.
 *
 * This handles lengths that are not more than 8 bytes currently. We probably
 * need to copy some of the code from `datumSerialize` (which is used to
 * serialize datums for transfer to parallel workers) to serialize arbitrary
 * data into an arrow array.
 */
static NullableDatum
arrow_get_datum_fixlen(const ArrowArray *array, Oid typid, int16 typlen, uint16 index)
{
	/* Validate input parameters to prevent NULL pointer dereference */
	if (array == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("arrow array is NULL")));
	
	/* Note: buffers[0] (validity) may be NULL per Arrow spec */
	if (array->buffers == NULL || array->buffers[1] == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("arrow array buffers are NULL or incomplete")));
	
	/* Validate index bounds */
	if (index >= array->length)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("array index %u out of bounds, array length is %lld",
						index, (long long) array->length)));
	
	const uint64 *restrict validity = array->buffers[0];
	const char *restrict values = array->buffers[1];
	const ArrowPrivate *apriv = arrow_private_get(array);
	Datum datum;

	Assert(typlen > 0);

	/* Check if validity bitmap is NULL - if so, all values are valid by Arrow spec */
	if (validity != NULL && !arrow_row_is_valid(validity, index))
		return (NullableDatum){ .isnull = true };

	/* Check for potential integer overflow in index * typlen calculation */
	if (typlen > 0 && (size_t) index > SIZE_MAX / (size_t) typlen)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("array index %u with typlen %d would overflow size_t",
						index, typlen)));

	/* Calculate buffer position safely */
	const size_t buffer_offset = (size_t) index * typlen;
	
	/* For fixed-length arrays, buffer size = array_length * typlen */
	const size_t expected_buffer_size = (size_t) array->length * typlen;
	
	/* Validate buffer bounds */
	if (buffer_offset + typlen > expected_buffer_size)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("buffer access [%zu, %zu) exceeds expected buffer size %zu",
						buffer_offset, buffer_offset + typlen, expected_buffer_size)));

	if (typid == BOOLOID)
	{
		/* Boolean type is handled differently from other fixed-length
		 * types. Booleans are stored as bitmap rather than as a fixed length
		 * type. For booleans, we need to check bit-level bounds */
		const size_t bit_index = index;
		const size_t byte_index = bit_index / 8;
		const size_t expected_bitmap_size = (array->length + 7) / 8; /* Round up to bytes */
		
		if (byte_index >= expected_bitmap_size)
			ereport(ERROR,
					(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
					 errmsg("boolean bit index %zu exceeds bitmap size %zu bytes",
							bit_index, expected_bitmap_size)));
		
		/* For boolean arrays, data is stored in buffer[1] as a bitmap */
		if (array->buffers[1] == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("boolean data buffer is NULL")));
		
		datum = BoolGetDatum(arrow_row_is_valid(array->buffers[1], index));
	}
	else
	{
		/* In order to handle fixed-length values of arbitrary size that are byref
		 * and byval, we use fetch_all() rather than rolling our own. This is
		 * taken from utils/adt/rangetypes.c */
		datum = ts_fetch_att(&values[buffer_offset], apriv->typbyval, typlen);
	}

	TS_DEBUG_LOG("retrieved fixlen value %s row %u from offset %u"
				 " in memory context %s",
				 datum_as_string(typid, datum, false),
				 index,
				 typlen * index,
				 GetMemoryChunkContext((void *) values)->name);

	return (NullableDatum){ .isnull = false, .value = datum };
}

NullableDatum
arrow_get_datum(const ArrowArray *array, Oid typid, int16 typlen, uint16 index)
{
	if (typlen == TYPLEN_VARLEN)
		return arrow_get_datum_varlen(array, typid, index);
	else
		return arrow_get_datum_fixlen(array, typid, typlen, index);
}

/*
 * Create an arrow array with memory for buffers.
 *
 * The space for buffers are allocated after the main structure.
 */
ArrowArray *
arrow_create_with_buffers(MemoryContext mcxt, int n_buffers)
{
	struct
	{
		ArrowArray array;
		const void *buffers[FLEXIBLE_ARRAY_MEMBER];
	} *array_with_buffers =
		MemoryContextAllocZero(mcxt, sizeof(ArrowArray) + (sizeof(const void *) * n_buffers));

	ArrowArray *array = &array_with_buffers->array;

	array->n_buffers = n_buffers;
	array->buffers = array_with_buffers->buffers;

	return array;
}
