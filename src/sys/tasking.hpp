#ifndef __PF_TASKING_HPP__
#define __PF_TASKING_HPP__

#include "sys/ref.hpp"
#include "sys/atomic.hpp"

#define PF_TASK_USE_DEDICATED_ALLOCATOR 1

namespace pf {

  /*! A task with a higher priority will be preferred to a task with a lower
   * priority. Note that the system does not completely comply with priorities.
   * Basically, because the system is distributed, it is possible that one
   * particular worker thread processes a low priority task while another thread
   * actually has higher priority tasks currently available
   */
  enum TaskPriority {
    CRITICAL_PRIORITY = 0u,
    HIGH_PRIORITY     = 1u,
    NORMAL_PRIORITY   = 2u,
    LOW_PRIORITY      = 3u,
    NUM_PRIORITY      = 4u,
    INVALID_PRIORITY  = 0xffffu
  };

  /*! Interface for all tasks handled by the tasking system */
  class Task : public RefCount
  {
  public:
    /*! It can complete one task and can be continued by one other task */
    INLINE Task(const char *taskName = NULL) :
      //name(taskName),
      toStart(1), toEnd(1),
      priority(NORMAL_PRIORITY), affinity(0xffff) {}
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
    /*! Set / get task priority and affinity */
    INLINE void setPriority(TaskPriority prio)  { this->priority = prio; }
    INLINE void setAffinity(int32 affi)         { this->affinity = affi; }
    INLINE TaskPriority getPriority(void) const { return this->priority; }
    INLINE uint16 getAffinity(void)       const { return this->affinity; }

#if PF_TASK_USE_DEDICATED_ALLOCATOR
    /*! Tasks use a scalable fixed size allocator */
    void* operator new(size_t size);
    /*! Deallocations go through the dedicated allocator too */
    void operator delete(void* ptr);
#endif /* PF_TASK_USE_DEDICATED_ALLOCATOR */

  private:
    friend class TaskSet;       //!< Will tweak the ending criterium
    friend class TaskScheduler; //!< Needs to access everything
    Ref<Task> toBeEnded;        //!< Signals it when finishing
    Ref<Task> toBeStarted;      //!< Triggers it when ready
    //const char *name;           //!< Debug facility mostly
    Atomic32 toStart;           //!< MBZ before starting
    Atomic32 toEnd;             //!< MBZ before ending
    TaskPriority priority;      //!< Task priority
    int16 affinity;             //!< The task will run on a particular thread
  };

  /*! Allow the run function to be executed several times */
  class TaskSet : public Task
  {
  public:
    /*! elemNum is the number of times to execute the run function */
    INLINE TaskSet(size_t elemNum, const char *name = NULL) :
      Task(name), elemNum(elemNum) {}
    /*! This function is user-specified */
    virtual void run(size_t elemID) = 0;

  private:
    virtual void run(void);  //!< Reimplemented for all task sets
    Atomic elemNum;          //!< Number of outstanding elements
  };

  /*! Mandatory before creating and running any task (MAIN THREAD) */
  void TaskingSystemStart(void);

  /*! Make the main thread enter the tasking system (MAIN THREAD) */
  void TaskingSystemEnd(void);

  /*! Make the main thread enter the tasking system (MAIN THREAD) */
  void TaskingSystemEnter(void);

  /*! Signal the *main* thread only to stop (THREAD SAFE) */
  void TaskingSystemInterruptMain(void);

  /*! Signal *all* threads to stop (THREAD SAFE) */
  void TaskingSystemInterrupt(void);

} /* namespace pf */

#endif /* __PF_TASKING_HPP__ */

