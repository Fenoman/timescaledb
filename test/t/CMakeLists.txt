if(CMAKE_BUILD_TYPE MATCHES Debug)
  list(APPEND PROVE_TEST_FILES 001_replication_telemetry.pl
       002_compressed_chunk_apache.pl)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

foreach(P_FILE ${PROVE_TEST_FILES})
  configure_file(${P_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${P_FILE} COPYONLY)
endforeach(P_FILE)

# extension_current_state() additionally checks pg_extension in assert builds.
set(LOADER_ASSERT_SCANS 0)
if(ASSERTIONS OR PG_USE_ASSERT_CHECKING)
  set(LOADER_ASSERT_SCANS 1)
endif()
configure_file(003_loader_cache.pl.in 003_loader_cache.pl @ONLY)
