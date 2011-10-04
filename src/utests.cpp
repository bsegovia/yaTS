#include "sys/tasking.hpp"
#include "sys/ref.hpp"
#include "sys/thread.hpp"
#include "sys/mutex.hpp"
#include "sys/sysinfo.hpp"

#define START_UTEST(TEST_NAME)                          \
void TEST_NAME(void)                                    \
{                                                       \
  std::cout << std::endl << "starting " <<              \
              #TEST_NAME << std::endl;

#define END_UTEST(TEST_NAME)                            \
  std::cout << "ending " << #TEST_NAME << std::endl;    \
}

using namespace pf;

///////////////////////////////////////////////////////////////////////////////
// Very simple test which basically does nothing
///////////////////////////////////////////////////////////////////////////////
class NothingTask : public Task {
public:
  virtual Task* run(void) { return NULL; }
};

class DoneTask : public Task {
public:
  virtual Task* run(void) { TaskingSystemInterrupt(); return NULL; }
};

START_UTEST(TestDummy)
  TaskingSystemStart();
  Task *done = NEW(DoneTask);
  Task *nothing = NEW(NothingTask);
  nothing->starts(done);
  done->scheduled();
  nothing->scheduled();
  TaskingSystemEnter();
  TaskingSystemEnd();
END_UTEST(TestDummy)

///////////////////////////////////////////////////////////////////////////////
// Simplest taskset test. An array is filled by each worker
///////////////////////////////////////////////////////////////////////////////
class SimpleTaskSet : public TaskSet {
public:
  INLINE SimpleTaskSet(size_t elemNum, uint32 *array_) :
    TaskSet(elemNum), array(array_) {}
  virtual void run(size_t elemID) { array[elemID] = 1u; }
  uint32 *array;
};

START_UTEST(TestTaskSet)
  const size_t elemNum = 1 << 20;
  TaskingSystemStart();
  uint32 *array = NEW_ARRAY(uint32, elemNum);
  for (size_t i = 0; i < elemNum; ++i) array[i] = 0;
  Task *done = NEW(DoneTask);
  Task *taskSet = NEW(SimpleTaskSet, elemNum, array);
  taskSet->starts(done);
  done->scheduled();
  taskSet->scheduled();
  TaskingSystemEnter();
  TaskingSystemEnd();
  for (size_t i = 0; i < elemNum; ++i)
    FATAL_IF(array[i] == 0, "TestTaskSet failed");
  DELETE_ARRAY(array);
END_UTEST(TestTaskSet)

///////////////////////////////////////////////////////////////////////////////
// We create a binary tree of tasks here. Each task spawn a two children upto a
// given maximum level. Then, a atomic value is updated per leaf. In that test,
// all tasks complete the ROOT directly
///////////////////////////////////////////////////////////////////////////////
enum { maxLevel = 20u };

/*! One node task per node in the tree. Task completes the root */
class NodeTask : public Task {
public:
  INLINE NodeTask(Atomic &value_, uint32 lvl_, Task *root_=NULL) :
    value(value_), lvl(lvl_) {
    this->root = root_ == NULL ? this : root_;
  }
  virtual Task* run(void);
  Atomic &value;
  Task *root;
  uint32 lvl;
};

Task* NodeTask::run(void) {
  if (this->lvl == maxLevel)
    this->value++;
  else {
    Task *left  = NEW(NodeTask, this->value, this->lvl+1, this->root);
    Task *right = NEW(NodeTask, this->value, this->lvl+1, this->root);
    left->ends(this->root);
    right->ends(this->root);
    left->scheduled();
    right->scheduled();
  }
  return NULL;
}

///////////////////////////////////////////////////////////////////////////////
// Same binary test as above but here each task completes its parent task
// directly. This stresses the completion system
///////////////////////////////////////////////////////////////////////////////

/*! One node task per node in the tree. Task completes its parent */
class CascadeNodeTask : public Task {
public:
  INLINE CascadeNodeTask(Atomic &value_, uint32 lvl_, Task *root_=NULL) :
    value(value_), lvl(lvl_) {}
  virtual Task* run(void);
  Atomic &value;
  uint32 lvl;
};

Task *CascadeNodeTask::run(void) {
  if (this->lvl == maxLevel)
    this->value++;
  else {
    Task *left  = NEW(CascadeNodeTask, this->value, this->lvl+1);
    Task *right = NEW(CascadeNodeTask, this->value, this->lvl+1);
    left->ends(this);
    right->ends(this);
    left->scheduled();
    right->scheduled();
  }
  return NULL;
}

/*! For both tree tests */
template<typename NodeType>
START_UTEST(TestTree)
  TaskingSystemStart();
  Atomic value(0u);
  std::cout << "nodeNum = " << (2 << maxLevel) - 1 << std::endl;
  double t = getSeconds();
  Task *done = NEW(DoneTask);
  Task *root = NEW(NodeType, value, 0);
  root->starts(done);
  done->scheduled();
  root->scheduled();
  TaskingSystemEnter();
  t = getSeconds() - t;
  std::cout << t * 1000. << " ms" << std::endl;
  TaskingSystemEnd();
  FATAL_IF(value != (1 << maxLevel), "TestTree failed");
END_UTEST(TestTree)

///////////////////////////////////////////////////////////////////////////////
// We try to stress the internal allocator here
///////////////////////////////////////////////////////////////////////////////
class AllocateTask : public TaskSet {
public:
  AllocateTask(size_t elemNum) : TaskSet(elemNum) {}
  virtual void run(size_t elemID);
  enum { allocNum = 1 << 10 };
  enum { iterNum = 1 << 5 };
};

void AllocateTask::run(size_t elemID) {
  Task *tasks[allocNum];
  for (int j = 0; j < iterNum; ++j) {
    const int taskNum = rand() % allocNum;
    for (int i = 0; i < taskNum; ++i) tasks[i] = NEW(NothingTask);
    for (int i = 0; i < taskNum; ++i) DELETE(tasks[i]);
  }
}

START_UTEST(TestAllocator)
  TaskingSystemStart();
  Task *done = NEW(DoneTask);
  Task *allocate = NEW(AllocateTask, 1 << 10);
  allocate->starts(done);
  done->scheduled();
  allocate->scheduled();
  TaskingSystemEnter();
  TaskingSystemEnd();
END_UTEST(TestAllocator)

///////////////////////////////////////////////////////////////////////////////
// We are making the queue full to make the system recurse to empty it
///////////////////////////////////////////////////////////////////////////////
class FullTask : public Task {
public:
  enum { taskToSpawn = 1u << 16u };
  FullTask(const char *name, Atomic &counter, int lvl = 0) :
    Task(name), counter(counter), lvl(lvl) {}
  ~FullTask(void) { lvl = 0xdead; }
  virtual Task* run(void) {
    if (lvl == 0)
      for (size_t i = 0; i < taskToSpawn; ++i) {
        Task *task = NEW(FullTask, "FullTaskLvl1", counter, 1);
        task->ends(this);
        task->scheduled();
      }
    else
      counter++;
    return NULL;
  }
  Atomic &counter;
  int lvl;
};

START_UTEST(TestFullQueue)
  Atomic counter(0u);
  TaskingSystemStart();
  Task *done = NEW(DoneTask);
  for (size_t i = 0; i < 64; ++i) {
    Task *task = NEW(FullTask, "FullTask", counter);
    task->starts(done);
    task->scheduled();
  }
  done->scheduled();
  TaskingSystemEnter();
  TaskingSystemEnd();
  FATAL_IF (counter != 64 * FullTask::taskToSpawn, "TestFullQueue failed");
END_UTEST(TestFullQueue)

int main(int argc, char **argv)
{
  startMemoryDebugger();

  TestDummy();
  TestTree<NodeTask>();
  TestTree<CascadeNodeTask>();
  TestTaskSet();
  TestAllocator();
  TestFullQueue();

  dumpAlloc();
  endMemoryDebugger();
  return 0;
}

