include (FindGLUT)
include_directories (${GLUT_INCLUDE_PATH})
set (SRC
  utests.cpp
  sys/filename.cpp
  sys/library.cpp
  sys/thread.cpp
  sys/alloc.cpp
  sys/tasking.cpp
  sys/sysinfo.cpp
  sys/mutex.cpp
  sys/condition.cpp
  sys/platform.cpp
  sys/string.cpp)
include_directories (.)
add_executable (app ${SRC})
if (UNIX)
  target_link_libraries(app ${GLUT_LIBRARY} pthread)
else (UNIX)
  target_link_libraries(app ${GLUT_LIBRARY})
endif (UNIX)

