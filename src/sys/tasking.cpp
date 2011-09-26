#include "sys/tasking.hpp"
#include "sys/ref.hpp"
#include "sys/thread.hpp"
#include "sys/mutex.hpp"
#include "sys/sysinfo.hpp"

#include <vector>

#define PF_TASK_STATICTICS 0
#if PF_TASK_STATICTICS
#define IF_TASK_STATISTICS(EXPR) do { EXPR; } while (0)
#else
#define IF_TASK_STATISTICS(EXPR) do { } while(0)
#endif

namespace pf {
  class Task;         // Basically an asynchronous function with dependencies
  class TaskSet;      // Idem but can be run N times
  class TaskAllocator;// Dedicated to allocate tasks and task sets

  /*! Structure used for work stealing */
  template <int elemNum>
  class TaskQueue
  {
  public:
    INLINE TaskQueue(void) :
#if PF_TASK_STATICTICS
      statInsertNum(0), statGetNum(0), statStealNum(0),
#endif
      head(0), tail(0)
    {}
    INLINE void insert(Task &task) {
      assert(head - tail < elemNum);
      tasks[head % elemNum] = &task;
      head++;
      IF_TASK_STATISTICS(statInsertNum++);
    }
    INLINE Ref<Task> get(void) {
      if (head == tail) return NULL;
      Lock<MutexType> lock(mutex);
      if (head == tail) return NULL;
      head--;
      Ref<Task> task = tasks[head % elemNum];
      tasks[head % elemNum] = NULL;
      IF_TASK_STATISTICS(statGetNum++);
      return task;
    }
    INLINE Ref<Task> steal(void) {
      if (head == tail) return NULL;
      Lock<MutexType> lock(mutex);
      if (head == tail) return NULL;
      Ref<Task> stolen = tasks[tail % elemNum];
      tasks[tail % elemNum] = NULL;
      tail++;
      IF_TASK_STATISTICS(statStealNum++);
      return stolen;
    }
#if PF_TASK_STATICTICS
    void printStats(void) {
      std::cout << "insertNum " << statInsertNum <<
                   ", getNum " << statGetNum <<
                   ", stealNum " << statStealNum << std::endl;
    }
    Atomic statInsertNum, statGetNum, statStealNum;
#endif
  private:
    typedef MutexActive MutexType;
    Ref<Task> tasks[elemNum];     //!< All tasks currently stored
    volatile atomic_t head, tail; //!< Current queue property
    MutexType mutex;             //!< Not lock-free right now
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
    /*! Number of threads running in the scheduler (not including main) */
    INLINE uint32 getThreadNum(void) { return this->threadNum; }

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
    INLINE void schedule(Task &task);

    friend class Task;            //!< Tasks ...
    friend class TaskSet;         // ... task sets ...
    friend class TaskAllocator;   // ... task allocator use the tasking system
    static THREAD uint32 threadID;//!< ThreadID for each thread
    enum { queueSize = 2048 };    //!< Number of task per queue
    TaskQueue<queueSize> *queues; //!< One queue per thread
    thread_t *threads;            //!< All threads currently running
    size_t threadNum;             //!< Total number of threads running
    size_t queueNum;              //!< Number of queues (should be threadNum+1)
    volatile bool dead;           //!< The tasking system should quit
  };

  /*! Allocator per thread */
  class ThreadStorage
  {
  public:
    ThreadStorage(void) :
#if PF_TASK_STATICTICS
      statNewChunkNum(0), statPushGlobalNum(0), statPopGlobalNum(0),
      statAllocateNum(0), statDeallocateNum(0),
#endif
      allocator(NULL)
    {
      for (size_t i = 0; i < maxHeap; ++i) {
        this->chunk[i] = NULL;
        this->currSize[i] = 0u;
      }
    }
    ~ThreadStorage(void) {
      for (size_t i = 0; i < toFree.size(); ++i) ALIGNED_FREE(toFree[i]);
    }

    /*! Will try to allocate from the local storage. Use std::malloc to
     *  allocate a new local chunk
     */
    INLINE void *allocate(size_t sz);

    /*! Free a task and put it in a free list. If too many tasks are
     *  deallocated, return a piece of it to the global heap
     */
    INLINE void deallocate(void *ptr);

    /*! Create a free list and store chunk information */
    void newChunk(uint32 chunkID);
    /*! Push back a group of tasks in the global heap */
    void pushGlobal(uint32 chunkID);
    /*! Pop a group of tasks from the global heap (if none, return NULL) */
    void popGlobal(uint32 chunkID);

#if PF_TASK_STATICTICS
    void printStats(void) {
      std::cout << "newChunkNum " << statNewChunkNum <<
                   ", pushGlobalNum  " << statPushGlobalNum <<
                   ", popGlobalNum  " << statPushGlobalNum <<
                   ", allocateNum  " << statAllocateNum <<
                   ", deallocateNum  " << statDeallocateNum << std::endl;
    }
    Atomic statNewChunkNum, statPushGlobalNum, statPopGlobalNum;
    Atomic statAllocateNum, statDeallocateNum;
#endif

  private:
    friend class TaskAllocator;
    enum { logChunkSize = 20 };           //!< log2(4KB)
    enum { chunkSize = 1<< logChunkSize}; //!< 4KB when taking memory from std
    enum { maxHeap = 10u };      //!< One heap per size (only power of 2)
    TaskAllocator *allocator;    //!< Handles global heap
    void *chunk[maxHeap];        //!< One heap per size
    uint32_t currSize[maxHeap];  //!< Sum of the free task sizes
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
    TaskAllocator(uint32 threadNum);
    ~TaskAllocator(void);
    void *allocate(size_t sz);
    void deallocate(void *ptr);
    enum { maxHeap = ThreadStorage::maxHeap };
    enum { maxSize = 1 << maxHeap };
    ThreadStorage *local;  //!< Local heaps (per thread and per size)
    void *global[maxHeap]; //!< Global heap shared by all threads
    MutexActive mutex;     //!< To protect the global heap
    uint32 threadNum;      //!< One thread storage per thread
  };

  TaskAllocator::TaskAllocator(uint32 threadNum_) : threadNum(threadNum_) {
    this->local = NEW_ARRAY(ThreadStorage, threadNum);
    for (size_t i = 0; i < threadNum; ++i) this->local[i].allocator = this;
    for (size_t i = 0; i < maxHeap; ++i) this->global[i] = NULL;
  }
  TaskAllocator::~TaskAllocator(void) {
#if PF_TASK_STATICTICS
    for (size_t i = 0; i < threadNum; ++i) this->local[i].printStats();
#endif
    DELETE_ARRAY(this->local);
  }

  void *TaskAllocator::allocate(size_t sz) {
    FATAL_IF (sz > maxSize, "Task size is too large (TODO remove that)");
    // We use free list for the task. Each free list node can be made of:
    // [pointer_to_next_node,pointer_to_next_chunk,sizeof(chunk)]
    // We therefore need three times the size of a pointer for the nodes
    // and therefore for the task
    sz = std::max(3 * sizeof(void*), sz);
    return this->local[TaskScheduler::threadID].allocate(sz);
  }

  void TaskAllocator::deallocate(void *ptr) {
    return this->local[TaskScheduler::threadID].deallocate(ptr);
  }

  void ThreadStorage::newChunk(uint32 chunkID) {
    IF_TASK_STATISTICS(statNewChunkNum++);
    // We store the size of the elements in the chunk header
    const uint32 elemSize = 1 << chunkID;
    char *chunk = (char *) ALIGNED_MALLOC(chunkSize, chunkSize);

    // We store this pointer to free it later while deleting the task
    // allocator
    this->toFree.push_back(chunk);
    *(uint32 *) chunk = elemSize;

    // Fill the free list here
    this->currSize[chunkID] = elemSize;
    char *data = (char*) chunk + CACHE_LINE;
    const char *end = (char*) chunk + chunkSize;
    *(void**) data = NULL; // Last element of the list is the first in chunk
    void *pred = data;
    data += elemSize;
    while (data < end) {
      *(void**) data = pred;
      pred = data;
      data += elemSize;
      this->currSize[chunkID] += elemSize;
    }
    this->chunk[chunkID] = pred;
  }

  void ThreadStorage::pushGlobal(uint32 chunkID) {
    IF_TASK_STATISTICS(statPushGlobalNum++);

    const uint32 elemSize = 1 << chunkID;
    void *list = this->chunk[chunkID];
    void *succ = list, *pred = NULL;
    while (this->currSize[chunkID] > chunkSize) {
      assert(succ);
      pred = succ;
      succ = *(void**) succ;
      this->currSize[chunkID] -= elemSize;
    }

    // If we pull off some nodes, then we push them back to the global heap
    if (pred) {
      *(void**) pred = NULL;
      this->chunk[chunkID] = succ;
      Lock<MutexActive> lock(allocator->mutex);
      ((void**) list)[1] = allocator->global[chunkID];
      allocator->global[chunkID] = list;
    }
  }

  void ThreadStorage::popGlobal(uint32 chunkID) {
    IF_TASK_STATISTICS(statPopGlobalNum++);
    void *list = NULL;
    assert(this->chunk[chunkID] == NULL);
    if (allocator->global[chunkID] == NULL) return;

    // Limit the contention time
    do {
      Lock<MutexActive> lock(allocator->mutex);
      list = allocator->global[chunkID];
      if (list == NULL) return;
      allocator->global[chunkID] = ((void**) list)[1];
    } while (0);

    // This is our new chunk
    this->chunk[chunkID] = list;
  }

  void* ThreadStorage::allocate(size_t sz) {
    IF_TASK_STATISTICS(statAllocateNum++);
    const uint32 chunkID = __bsf(int(nextHighestPowerOf2(uint32(sz))));
    if (UNLIKELY(this->chunk[chunkID] == NULL)) {
      this->popGlobal(chunkID);
      if (UNLIKELY(this->chunk[chunkID] == NULL))
        this->newChunk(chunkID);
    }
    void *curr = this->chunk[chunkID];
    this->chunk[chunkID] = *(void**) curr; // points to its predecessor
    this->currSize[chunkID] -= 1 << chunkID;
    return curr;
  }

  void ThreadStorage::deallocate(void *ptr) {
    IF_TASK_STATISTICS(statDeallocateNum++);
    // Figure out with the chunk header the size of this element
    char *chunk = (char*) (uintptr_t(ptr) & ~((1<<logChunkSize)-1));
    const uint32 elemSize = *(uint32*) chunk;
    const uint32 chunkID = __bsf(int(nextHighestPowerOf2(uint32(elemSize))));

    // Insert the free element in the free list
    void *succ = this->chunk[chunkID];
    *(void**) ptr = succ;
    this->chunk[chunkID] = ptr;
    this->currSize[chunkID] += elemSize;

    // If this thread has too many free tasks, we give some to the global heap
    if (this->currSize[chunkID] > 2 * chunkSize)
      this->pushGlobal(chunkID);
  }

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
#if PF_TASK_STATICTICS
    for (size_t i = 0; i < queueNum; ++i) {
      std::cout << "Task Queue " << i << " ";
      queues[i].printStats();
    }
#endif
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
      Ref<Task> task = This->queues[thread->tid].get();
      while (!task) {
        for (size_t i = 0; i < thread->tid; ++i)
          if (task = This->queues[i].steal()) break;
        if (!task)
          for (size_t i = threadID+1; i < This->queueNum; ++i)
            if (task = This->queues[i].steal()) break;
        if (UNLIKELY(This->dead)) goto end;
      }
      if (UNLIKELY(This->dead)) break;

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
  end:
    DELETE(thread);
  }

  void TaskScheduler::go(void) {
    Thread *thread = NEW(Thread, 0, *this);
    threadFunction(thread);
  }

  static TaskScheduler *scheduler = NULL;
  static TaskAllocator *allocator = NULL;

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

#if PF_TASK_USE_DEDICATED_ALLOCATOR
  void *Task::operator new(size_t size) { return allocator->allocate(size); }
  void Task::operator delete(void *ptr) { allocator->deallocate(ptr); }
#endif

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
    atomic_t curr;
    if (this->elemNum > 2) {
      this->toEnd += 2;
      scheduler->schedule(*this);
      scheduler->schedule(*this);
      while ((curr = --this->elemNum) >= 0)
        this->run(curr);
    } else if (this->elemNum > 1) {
      this->toEnd++;
      scheduler->schedule(*this);
      while ((curr = --this->elemNum) >= 0)
        this->run(curr);
    } else if (--this->elemNum == 0)
      this->run(0);
  }

  void startTaskingSystem(void) {
    FATAL_IF (scheduler != NULL, "scheduler is already running");
    scheduler = NEW(TaskScheduler);
    allocator = NEW(TaskAllocator, scheduler->getThreadNum()+1);
#if 0
    void *ptr[1000];
    for (int i = 0; i < 1000; ++i)
      ptr[i] = allocator->allocate(CACHE_LINE);
    for (int i = 0; i < 1000; ++i)
      allocator->deallocate(ptr[i]);
    for (int i = 0; i < 1000; ++i)
      ptr[i] = allocator->allocate(CACHE_LINE);
#endif
  }

  void enterTaskingSystem(void) {
    FATAL_IF (scheduler == NULL, "scheduler not started");
    scheduler->go();
  }

  void endTaskingSytem(void) {
    SAFE_DELETE(scheduler);
    SAFE_DELETE(allocator);
    scheduler = NULL;
    allocator = NULL;
  }

  void interruptTaskingSystem(void) {
    FATAL_IF (scheduler == NULL, "scheduler not started");
    scheduler->die();
  }
}

#undef PF_TASK_STATICTICS
#undef IF_TASK_STATISTICS

