#ifndef __PF_TASKING_HPP__
#define __PF_TASKING_HPP__

#include "sys/ref.hpp"
#include "sys/atomic.hpp"

#define PF_TASK_USE_DEDICATED_ALLOCATOR 1

namespace pf {

  /*! Interface for all tasks handled by the tasking system */
  class Task : public RefCount
  {
  public:
    /*! It can complete one task and can be continued by one other task */
    INLINE Task(Task *completion_, Task *continuation_) :
      completion(completion_),
      continuation(continuation_),
      toStart(1), toEnd(1) {
      if (continuation) continuation->toStart++;
      if (completion) completion->toEnd++;
    }
    /*! To override while specifying a task */
    virtual void run(void) = 0;
    /*! Now the task is built and immutable */
    void done(void);
#if PF_TASK_USE_DEDICATED_ALLOCATOR
    /*! Tasks use a scalable fixed allocator */
    void* operator new(size_t size);
    /*! Deallocations go through the dedicated allocator too */
    void operator delete(void* ptr);
#endif

  private:
    friend class TaskSet;       //!< Will tweak the ending criterium
    friend class TaskScheduler; //!< Needs to access everything
    Ref<Task> completion;       //!< Signalled it when finishing
    Ref<Task> continuation;     //!< Triggers it when ready
    Atomic toStart;             //!< MBZ before starting
    Atomic toEnd;               //!< MBZ before ending
  };

  /*! Allow the run function to be executed several times */
  class TaskSet : public Task
  {
  public:
    /*! As for Task, it has both completion and continuation */
    TaskSet(size_t elemNum, Task *completion = NULL, Task *continuation = NULL);
    /*! This function is user-specified */
    virtual void run(size_t elemID) = 0;

  private:
    virtual void run(void);  //!< Reimplemented for all task sets
    Atomic elemNum;          //!< Number of outstanding elements
  };

  /*! Mandatory before creating and running any task (MAIN THREAD) */
  void startTaskingSystem(void);

  /*! Make the main thread enter the tasking system (MAIN THREAD) */
  void enterTaskingSystem(void);

  /*! Cleanly deallocate and shutdown everything (MAIN THREAD) */
  void endTaskingSytem(void);

  /*! Basically signal all threads to stop (THREAD SAFE) */
  void interruptTaskingSystem(void);

} /* namespace pf */

#endif /* __PF_TASKING_HPP__ */

