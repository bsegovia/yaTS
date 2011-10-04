#ifndef __PF_TASKING_HPP__
#define __PF_TASKING_HPP__

#include "sys/ref.hpp"
#include "sys/atomic.hpp"

#define PF_TASK_USE_DEDICATED_ALLOCATOR 1

namespace pf {

  /*! A task with a higher priority will be preferred to a task with a lower
   *  priority. Note that the system does not completely comply with
   *  priorities. Basically, because the system is distributed, it is possible
   *  that one particular worker thread processes a low priority task while
   *  another thread actually has higher priority tasks currently available
   */
  struct TaskPriority {
    enum {
      CRITICAL = 0u,
      HIGH     = 1u,
      NORMAL   = 2u,
      LOW      = 3u,
      NUM      = 4u,
      INVALID  = 0xffffu
    };
  };

  /*! Describe the current state of a task. This is only used in DEBUG mode to
   *  assert the correctness of the operations (like Task::starts or Task::ends
   *  which only operates on tasks with specific states). To be honest, this is a
   *  bit bullshit code. I think using different proxy types for tasks based on
   *  their state is the way to go. This would enforce correctness of the code
   *  through the typing system which is just better since static. Anyway.
   */
#ifndef NDEBUG
  struct TaskState {
    enum {
      NEW       = 0u,
      READY     = 2u,
      RUNNING   = 3u,
      DONE      = 4u,
      NUM       = 5u,
      INVALID   = 0xffffu
    };
  };
#endif /* NDEBUG */

  /*! Interface for all tasks handled by the tasking system */
  class Task : public RefCount
  {
  public:
    /*! It can complete one task and can be continued by one other task */
    INLINE Task(const char *taskName = NULL) :
      name(taskName),
      toStart(1), toEnd(1),
      priority(uint16(TaskPriority::NORMAL)), affinity(0xffffu)
#ifndef NDEBUG
      , state(uint16(TaskState::NEW))
#endif
    {
      // The scheduler will remove this reference
      this->refInc();
    }
    /*! To override while specifying a task. This is basically the code to
     * execute. The user can optionally return a task which will by-pass the
     * scheduler and will run *immediately* after this one. This is a classical
     * continuation passing style strategy when we have a depth first scheduling
     */
    virtual Task* run(void) = 0;
    /*! Task is built and will be ready when all start dependencies are over */
    void scheduled(void);
    /*! The given task cannot *start* as long as "other" is not complete */
    INLINE void starts(Task *other) {
      if (UNLIKELY(other == NULL)) return;
      assert(other->state == TaskState::NEW);
      if (UNLIKELY(this->toBeStarted)) return; // already a task to start
      other->toStart++;
      this->toBeStarted = other;
    }
    /*! The given task cannot *end* as long as "other" is not complete */
    INLINE void ends(Task *other) {
      if (UNLIKELY(other == NULL)) return;
      assert(other->state == TaskState::NEW ||
             other->state == TaskState::RUNNING);
      if (UNLIKELY(this->toBeEnded)) return;  // already a task to end
      other->toEnd++;
      this->toBeEnded = other;
    }
    /*! Set / get task priority and affinity */
    INLINE void setPriority(uint16 prio) {
      assert(this->state == TaskState::NEW);
      this->priority = prio;
    }
    INLINE void setAffinity(uint16 affi) {
      assert(this->state == TaskState::NEW);
      this->affinity = affi;
    }
    INLINE uint16 getPriority(void) const { return this->priority; }
    INLINE uint16 getAffinity(void) const { return this->affinity; }

#if PF_TASK_USE_DEDICATED_ALLOCATOR
    /*! Tasks use a scalable fixed size allocator */
    void* operator new(size_t size);
    /*! Deallocations go through the dedicated allocator too */
    void operator delete(void* ptr);
#endif /* PF_TASK_USE_DEDICATED_ALLOCATOR */

  private:
    friend class TaskSet;        //!< Will tweak the ending criterium
    friend class TaskScheduler;  //!< Needs to access everything
    Ref<Task> toBeEnded;         //!< Signals it when finishing
    Ref<Task> toBeStarted;       //!< Triggers it when ready
    const char *name;            //!< Debug facility mostly
    Atomic32 toStart;            //!< MBZ before starting
    Atomic32 toEnd;              //!< MBZ before ending
    uint16 priority;             //!< Task priority
    uint16 affinity;             //!< The task will run on a particular thread
  public: IF_DEBUG(uint16 state);//!< Will assert correctness of the operations
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
    virtual Task* run(void);  //!< Reimplemented for all task sets
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

