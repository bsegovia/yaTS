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
    INLINE Task(void) : toStart(1), toEnd(1) {}
    /*! To override while specifying a task */
    virtual void run(void) = 0;
    /*! Now the task is built and is allowed to be scheduled */
    void done(void);
    /*! The given task cannot *start* as long as this task is not done */
    INLINE void starts(Task *other) {
      if (UNLIKELY(other == NULL)) return;
      assert(this->toBeStarted == false);
      other->toStart++;
      this->toBeStarted = other;
    }
    /*! The given task cannot *end* as long as this task is not done */
    INLINE void ends(Task *other) {
      if (UNLIKELY(other == NULL)) return;
      assert(this->toBeEnded == false);
      other->toEnd++;
      this->toBeEnded = other;
    }

#if PF_TASK_USE_DEDICATED_ALLOCATOR
    /*! Tasks use a scalable fixed allocator */
    void* operator new(size_t size);
    /*! Deallocations go through the dedicated allocator too */
    void operator delete(void* ptr);
#endif /* PF_TASK_USE_DEDICATED_ALLOCATOR */

  private:
    friend class TaskSet;       //!< Will tweak the ending criterium
    friend class TaskScheduler; //!< Needs to access everything
    Ref<Task> toBeEnded;        //!< Signals it when finishing
    Ref<Task> toBeStarted;      //!< Triggers it when ready
    Atomic toStart;             //!< MBZ before starting
    Atomic toEnd;               //!< MBZ before ending
  };

  /*! Allow the run function to be executed several times */
  class TaskSet : public Task
  {
  public:
    /*! elemNum is the number of times to execute the run function */
    INLINE TaskSet(size_t elemNum) : elemNum(elemNum) {}
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

