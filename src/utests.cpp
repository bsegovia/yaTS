#include "sys/tasking.hpp"
#include "sys/ref.hpp"
#include "sys/thread.hpp"
#include "sys/mutex.hpp"
#include "sys/sysinfo.hpp"

using namespace pf;

///////////////////////////////////////////////////////////////////////////////
// Very simple test which basically does nothing
///////////////////////////////////////////////////////////////////////////////
class NothingTask : public Task {
public:
  INLINE NothingTask(Task *completion = NULL, Task *continuation = NULL) :
    Task(completion, continuation) {}
  virtual void run(void) { }
};

class DoneTask : public Task {
public:
  INLINE DoneTask(Task *completion = NULL, Task *continuation = NULL) :
    Task(completion, continuation) {}
  virtual void run(void) { shutdownTaskingSystem(); }
};

void dummyTest(void)
{
  startTaskingSystem();
  Task *done = NEW(DoneTask);
  Task *nothing = NEW(NothingTask, NULL, done);
  done->done();
  nothing->done();
  enterTaskingSystem();
  endTaskingSytem();
}

///////////////////////////////////////////////////////////////////////////////
// Simplest taskset test. An array is filled by each worker
///////////////////////////////////////////////////////////////////////////////
class SimpleTaskSet : public TaskSet {
public:
  INLINE SimpleTaskSet(size_t elemNum, Task *continuation, uint32 *array_) :
    TaskSet(elemNum, NULL, continuation), array(array_) {}
  virtual void run(size_t elemID) {
    array[elemID] = 1u;
  }
  uint32 *array;
};

void taskSetTest(void)
{
  const size_t elemNum = 1024;
  startTaskingSystem();
  uint32 *array = NEW_ARRAY(uint32, elemNum);
  for (size_t i = 0; i < elemNum; ++i) array[i] = 0;
  Task *done = NEW(DoneTask);
  Task *taskSet = NEW(SimpleTaskSet, elemNum, done, array);
  done->done();
  taskSet->done();
  enterTaskingSystem();
  endTaskingSytem();
  for (size_t i = 0; i < elemNum; ++i)
    FATAL_IF(array[i] == 0, "taskSetTest failed");
  DELETE_ARRAY(array);
}

///////////////////////////////////////////////////////////////////////////////
// We create a binary tree of tasks here. Each task spawn a two children upto a
// given maximum level. Then, a atomic value is updated per leaf. In that test,
// all tasks complete the ROOT directly
///////////////////////////////////////////////////////////////////////////////
enum { maxLevel = 10u };

/*! One node task per node in the tree. Task completes the root */
class NodeTask : public Task {
public:
  INLINE NodeTask(Atomic &value_, uint32 lvl_, Task *root_=NULL, Task *cont_=NULL) :
    Task(root_, cont_), value(value_), lvl(lvl_) {
    this->root = root_ == NULL ? this : root_;
  }
  virtual void run(void);
  Atomic &value;
  Task *root;
  uint32 lvl;
};

void NodeTask::run(void) {
  if (this->lvl == maxLevel)
    this->value++;
  else {
    Task *left  = NEW(NodeTask, this->value, this->lvl+1, this->root);
    Task *right = NEW(NodeTask, this->value, this->lvl+1, this->root);
    left->done();
    right->done();
  }
}

///////////////////////////////////////////////////////////////////////////////
// Same binary test as above but here each task completes its parent task
// directly. This stresses the completion system
///////////////////////////////////////////////////////////////////////////////

/*! One node task per node in the tree. Task completes its parent */
class CascadeNodeTask : public Task {
public:
  INLINE CascadeNodeTask(Atomic &value_, uint32 lvl_, Task *root_=NULL, Task *cont_=NULL) :
    Task(root_, cont_), value(value_), lvl(lvl_) {}
  virtual void run(void);
  Atomic &value;
  uint32 lvl;
};

void CascadeNodeTask::run(void) {
  if (this->lvl == maxLevel)
    this->value++;
  else {
    Task *left  = NEW(NodeTask, this->value, this->lvl+1, this);
    Task *right = NEW(NodeTask, this->value, this->lvl+1, this);
    left->done();
    right->done();
  }
}

/*! For both tests */
template<typename NodeType>
void treeTest(void)
{
  startTaskingSystem();
  Atomic value(0u);
  Task *done = NEW(DoneTask);
  Task *root = NEW(NodeType, value, 0, NULL, done);
  done->done();
  root->done();
  enterTaskingSystem();
  endTaskingSytem();
  FATAL_IF(value != (1 << maxLevel), "treeTest failed");
}

int main(int argc, char **argv)
{
  startMemoryDebugger();
#if 0
  dummyTest();
  treeTest<NodeTask>();
  treeTest<CascadeNodeTask>();
#endif
  taskSetTest();
  endMemoryDebugger();
  return 0;
}

