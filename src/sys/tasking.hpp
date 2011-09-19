#ifndef __PF_TASKING_HPP__
#define __PF_TASKING_HPP__

#include "sys/ref.hpp"
#include "sys/atomic.hpp"

namespace pf {

  /*! Interface for all tasks handled by the tasking system */
  class Task : public RefCount
  {
  public:

    /*! It can complete one task and can be continued by one other task */
    Task(Task *completion_ = NULL, Task *continuation_ = NULL);

    /*! Now the task is built and immutable */
    void done(void);

    /*! To override while specifying a task */
    virtual void run(void) = 0;

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

  /*! Start to shutdown the tasking system (THREAD SAFE) */
  void shutdownTaskingSystem(void);

} /* namespace pf */

#endif /* __PF_TASKING_HPP__ */

