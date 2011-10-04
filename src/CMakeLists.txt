include_directories (${GLUT_INCLUDE_PATH})
if (PF_USE_BLOB)
  set (SRC blob.cpp)
else (PF_USE_BLOB)
  set (SRC
    utests.cpp
    sys/thread.cpp
    sys/alloc.cpp
    sys/tasking.cpp
    sys/sysinfo.cpp
    sys/mutex.cpp
    sys/condition.cpp
    sys/platform.cpp)
endif (PF_USE_BLOB)
include_directories (.)
add_executable (app ${SRC})
if (UNIX)
  target_link_libraries(app pthread)
else (UNIX)
  target_link_libraries(app)
endif (UNIX)

