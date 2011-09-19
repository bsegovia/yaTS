#include "sys/tasking.hpp"
#include "sys/ref.hpp"
#include "sys/thread.hpp"
#include "sys/mutex.hpp"
#include "sys/sysinfo.hpp"

#include <vector>

namespace pf {

  /*! Structure used for work stealing */
  template <int elemNum>
  class TaskQueue
  {
  public:
    INLINE TaskQueue(void) : head(0), tail(0) {}
    INLINE void insert(Task &task) {
      assert(atomic_t(head) - atomic_t(tail) < elemNum);
      tasks[atomic_t(head) % elemNum] = &task;
      head++;
    }
    INLINE Ref<Task> get(void) {
      if (head == tail) return NULL;
      Lock<MutexActive> lock(mutex);
      if (head == tail) return NULL;
      head--;
      Ref<Task> task = tasks[head % elemNum];
      tasks[head % elemNum] = NULL;
      return task;
    }
    INLINE Ref<Task> steal(void) {
      if (head == tail) return NULL;
      Lock<MutexActive> lock(mutex);
      if (head == tail) return NULL;
      Ref<Task> stolen = tasks[tail % elemNum];
      tasks[tail % elemNum] = NULL;
      tail++;
      return stolen;
    }
  private:
    Ref<Task> tasks[elemNum]; //!< All tasks currently stored
    Atomic head, tail;        //!< Current queue property
    MutexActive mutex;        //!< Not lock-free right now
  };

  /*! Handle the scheduling of all tasks. We basically implement here a
   *  work-stealing algorithm. Each thread has each own queue where it picks
   *  up task in depth first order. Each thread can also steal other tasks
   *  in breadth first order when his own queue is empty
   */
  class TaskScheduler {
  public:

    /*! If threadNum == 0, use the maximum number of threads */
    TaskScheduler(int threadNum_ = -1);
    ~TaskScheduler(void);

    /*! Call by the main thread to enter the tasking system */
    void go(void);

    /*! Interrupt all threads */
    INLINE void die(void) { dead = true; }

    /*! Data provided to each thread */
    struct Thread {
      Thread (size_t tid, TaskScheduler &scheduler_) :
        tid(tid), scheduler(scheduler_) {}
      size_t tid;
      TaskScheduler &scheduler;
    };

  private:

    /*! Function run by each thread */
    static void threadFunction(Thread *thread);
    /*! Schedule a task which is now ready to execute */
    void schedule(Task &task);

    friend class Task;            //!< Only tasks ...
    friend class TaskSet;         //   ... and task sets use the tasking system
    static THREAD uint32 threadID;//!< ThreadID for each thread
    enum { queueSize = 2048 };    //!< Number of task per queue
    TaskQueue<queueSize> *queues; //!< One queue per thread
    thread_t *threads;            //!< All threads currently running
    size_t threadNum;             //!< Total number of threads running
    size_t queueNum;              //!< Number of queues (should be threadNum+1)
    volatile bool dead;           //!< The tasking system should quit
  };

#if 0
  /*! Allocator per thread */
  struct ThreadStorage {
    ThreadStorage(void) {
      for (size_t i = 0; i < maxHeap; ++i) {
        this->data[i] = NULL;
        this->maxNum[i] = chunkSize / (1 << i);
        this->curr[i] = 0;
      }
    }
    ~ThreadStorage(void) {
      // TODO free the local heaps
    }
    enum { maxHeap = 10u };      //!< One heap per size (only power of 2)
    enum { chunkSize = 4 * KB }; //!< 4KB when taking memory from std
    void *data[maxHeap];         //!< One heap per size
    uint32_t maxNum[maxHeap];    //!< Maximum number of task in the heap
    uint32_t curr[maxHeap];      //!< Current number of free tasks in the heap
    std::vector<void*> toFree;   //!< All chunks allocated (per thread)
  };

  /*! TaskAllocator will speed up task allocation with fast dedicated thread
   *  local storage and fixed size allocation strategy. Each thread maintains
   *  its own list of free tasks. When empty, it first tries to get some tasks
   *  from the global task heap. If the global heap is empty, it just allocates
   *  a new pool of task with a std::malloc. If the local pool is "full", a
   *  chunk of tasks is pushed back into the global heap. Note that the task
   *  allocator is really a growing pool. We *never* give back the chunk of
   *  memory taken from std::malloc (except when the allocator is destroyed)
   */
  class TaskAllocator {
  public:
    /*! Constructor. Here this is the total number of threads using the pool (ie
     *  number of worker threads + main thread)
     */
    TaskAllocator(int threadNum);
    ~TaskAllocator(void);
    void *allocate(size_t sz);
    void deallocate(void *ptr, size_t sz);
  private:
    ThreadStorage *local;                  //!< Local heaps (per thread and per size)
    void *global[ThreadStorage::maxHeap]; //!< Global heap shared by all threads
    MutexActive mutex;                    //!< To protect the global heap
  };

  TaskAllocator::TaskAllocator(int threadNum)
  {
    this->local =  NEW_ARRAY(ThreadStorage, threadNum);
    for (size_t i = 0; i < maxHeap; ++i) this->global[i] = NULL;
  }

  TaskAllocator::allocate(size_t sz)
  {
    sz = nex
  }
#endif
  TaskScheduler::TaskScheduler(int threadNum_) :
    queues(NULL), threads(NULL), dead(false)
  {
    if (threadNum_ < 0) threadNum_ = getNumberOfLogicalThreads() - 1;
    this->threadNum = threadNum_;

    // We have a work queue for the main thread too
    this->queueNum = threadNum+1;
    this->queues = NEW_ARRAY(TaskQueue<queueSize>, queueNum);

    // Only if we have dedicated worker threads
    if (threadNum > 0) {
      this->threads = NEW_ARRAY(thread_t, threadNum);
      const size_t stackSize = 4*MB;
      for (size_t i = 0; i < threadNum; ++i) {
        const int affinity = int(i+1);
        Thread *thread = NEW(Thread,i+1,*this);
        thread_func threadFunc = (thread_func) threadFunction;
        threads[i] = createThread(threadFunc, thread, stackSize, affinity);
      }
    }
  }

  void TaskScheduler::schedule(Task &task) {
    queues[this->threadID].insert(task);
  }

  TaskScheduler::~TaskScheduler(void) {
    if (threads)
      for (size_t i = 0; i < threadNum; ++i)
        join(threads[i]);
    SAFE_DELETE_ARRAY(threads);
    SAFE_DELETE_ARRAY(queues);
  }

  THREAD uint32 TaskScheduler::threadID = 0;

  void TaskScheduler::threadFunction(TaskScheduler::Thread *thread)
  {
    threadID = thread->tid;
    TaskScheduler *This = &thread->scheduler;

    // We try to pick up a task from our queue and then we try to steal a task
    // from other queues
    for (;;) {
      Ref<Task> task = This->queues[threadID].get();
      while (!task && !This->dead) {
        for (size_t i = 0; i < threadID; ++i)
          if (task = This->queues[i].steal()) break;
        if (!task)
          for (size_t i = threadID+1; i < This->queueNum; ++i)
            if (task = This->queues[i].steal()) break;
      }
      if (This->dead) break;

      // Execute the function
      task->run();

      // Explore the completions and runs all continuations if any
      do {
        const atomic_t stillRunning = --task->toEnd;

        // We are done here
        if (stillRunning == 0) {
          // Run the continuation if any
          if (task->continuation) {
            task->continuation->toStart--;
            if (task->continuation->toStart == 0)
              This->schedule(*task->continuation);
          }
          // Traverse all completions to signal we are done
          task = task->completion;
        }
        else
          task = NULL;
      } while (task);
    }
    DELETE(thread);
  }

  void TaskScheduler::go(void) {
    Thread *thread = NEW(Thread, 0, *this);
    threadFunction(thread);
  }

  static TaskScheduler *scheduler = NULL;

  void Task::done(void) {
    this->toStart--;
    if (this->toStart == 0) scheduler->schedule(*this);
  }

  Task::Task(Task *completion_, Task *continuation_) :
    completion(completion_),
    continuation(continuation_),
    toStart(1), toEnd(1)
  {
    if (continuation) continuation->toStart++;
    if (completion) completion->toEnd++;
  }

  TaskSet::TaskSet(size_t elemNum_, Task *completion, Task *continuation) :
    Task(completion, continuation), elemNum(elemNum_) { }

  void TaskSet::run(void)
  {
    // The basic idea with task sets is to reschedule the task in its own
    // queue to have it stolen by another thread. Once done, we simply execute
    // the code of the task set run function concurrently with other threads.
    // The only downside of this approach is that the rescheduled task *must*
    // be picked up again to completely end the task set. This basically may
    // slightly delay the ending of it (note that much since we enqueue /
    // dequeue in LIFO style here)
    // Also, note that we reenqueue the task twice since it allows an
    // exponential propagation of the task sets in the other thread queues
    if (this->elemNum > 1) {
      this->toEnd += 2;
      scheduler->schedule(*this);
      scheduler->schedule(*this);
      atomic_t curr;
      while ((curr = --this->elemNum) >= 0)
        this->run(curr);
    }
    else if (--this->elemNum == 0)
      this->run(0);
  }

  void startTaskingSystem(void) {
    FATAL_IF (scheduler != NULL, "scheduler is already running");
    scheduler = NEW(TaskScheduler);
  }

  void enterTaskingSystem(void) {
    FATAL_IF (scheduler == NULL, "scheduler not started");
    scheduler->go();
  }

  void endTaskingSytem(void) {
    SAFE_DELETE(scheduler);
    scheduler = NULL;
  }

  void shutdownTaskingSystem(void) {
    FATAL_IF (scheduler == NULL, "scheduler not started");
    scheduler->die();
  }
}

