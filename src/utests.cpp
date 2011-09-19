#include "sys/tasking.hpp"
#include "sys/ref.hpp"
#include "sys/thread.hpp"
#include "sys/mutex.hpp"
#include "sys/sysinfo.hpp"

using namespace pf;

class NothingTask : public Task {
public:
  NothingTask(Task *completion = NULL, Task *continuation = NULL) :
    Task(completion, continuation) {}
  virtual void run(void) {
  }
};

class DoneTask : public Task {
public:
  DoneTask(Task *completion = NULL, Task *continuation = NULL) :
    Task(completion, continuation) {}
  virtual void run(void) { endTaskingSytem(); }
};

int main(int argc, char **argv)
{
  startMemoryDebugger();
  startTaskingSystem();
  Task *done = NEW(DoneTask);
  Task *nothing = NEW(NothingTask, NULL, done);
  done->done();
  nothing->done();
  enterTaskingSystem();

  endMemoryDebugger();
  return 0;
}

