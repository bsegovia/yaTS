#include "sys/tasking.hpp"
#include "sys/ref.hpp"
#include "sys/thread.hpp"
#include "sys/mutex.hpp"
#include "sys/sysinfo.hpp"

#include <vector>
#include <cstdlib>

/* One important remark here about reference counting. Tasks are referenced
 * counted but we do not use Ref<Task> here. This is for performance reasons.
 * We therefore *manually* handle the extra references the scheduler may have
 * on a task. This is nasty but maintains a reasonnable speed in the system.
 * Using Ref<Task> in the queue makes the system roughly twice slower
 */
#define PF_TASK_STATICTICS 0
#if PF_TASK_STATICTICS
#define IF_TASK_STATISTICS(EXPR) do { EXPR; } while (0)
#else
#define IF_TASK_STATISTICS(EXPR) do { } while(0)
#endif

namespace pf {
  ///////////////////////////////////////////////////////////////////////////
  /// Declaration of the internal classes of the tasking system
  ///////////////////////////////////////////////////////////////////////////

  class Task;          // Basically an asynchronous function with dependencies
  class TaskSet;       // Idem but can be run N times
  class TaskAllocator; // Dedicated to allocate tasks and task sets

  /*! Fast random number generator (TODO find something that works) */
  struct ALIGNED(CACHE_LINE) FastRand {
    FastRand(void) { x = rand(); y = rand(); z = rand(); }
    unsigned long rand(void) {
#if 0
      unsigned long t;
      x ^= x << 16;
      x ^= x >> 5;
      x ^= x << 1;
      t = x;
      x = y;
      y = z;
      z = t ^ x ^ y;
#else
      ++z;
#endif
      return z;
    }
    unsigned long x, y, z;
    ALIGNED_CLASS
  };

  /*! Structure used to issue ready-to-process tasks */
  template <int elemNum>
  struct ALIGNED(CACHE_LINE) TaskQueue
  {
  public:
    INLINE TaskQueue(void) {
      for (size_t i = 0; i < NUM_PRIORITY; ++i) head[i] = tail[i] = 0;
    }

  protected:
    /*! Return the bit mask of the four queues:
     *  - 1 if there is any task
     *  - 0 if empty
     *  Since we properly sort priorities from 0 to 3, using bit scan forward will
     *  return the first non-empty queue with the highest priority
     */
    INLINE int getActiveMask(void) const {
      const __m128i len = _mm_sub_epi32(tail.v, head.v);
      return _mm_movemask_ps(_mm_castsi128_ps(len));
    }

    Task* tasks[NUM_PRIORITY][elemNum]; //!< All tasks currently stored
    MutexActive mutex;                  //!< Not lock-free right now
    union {
      INLINE volatile int32& operator[] (int32 prio) { return x[prio]; }
      volatile int32 x[NUM_PRIORITY];
      volatile __m128i v;
    } head, tail;
  };

  /*! For work stealing:
   *  - only the owner (ie the victim) inserts tasks
   *  - the owner picks up tasks in depth first order (LIFO)
   *  - the stealers pick up tasks in breadth first order (FIFO)
   */
  template <int elemNum>
  struct TaskWorkStealingQueue : TaskQueue<elemNum> {
    TaskWorkStealingQueue(void)
#if PF_TASK_STATICTICS
      : statInsertNum(0), statGetNum(0), statStealNum(0)
#endif /* PF_TASK_STATICTICS */
    {}

    /*! No need to lock here since only the owner can push a task */
    INLINE void insert(Task &task);
    /*! Both stealers and the victim can pick up a task: we lock */
    INLINE Task* get(void);
    /*! Idem: we lock */
    INLINE Task* steal(void);

#if PF_TASK_STATICTICS
    void printStats(void) {
      std::cout << "insertNum " << statInsertNum <<
                   ", getNum " << statGetNum <<
                   ", stealNum " << statStealNum << std::endl;
    }
    Atomic32 statInsertNum, statGetNum, statStealNum;
#endif /* PF_TASK_STATICTICS */
  };

  /*! Tasks with affinity go here. For this queue:
   *  - any thread can push a task
   *  - only the owner can pick up tasks
   */
  template <int elemNum>
  struct TaskAffinityQueue : TaskQueue<elemNum> {
    TaskAffinityQueue (void)
#if PF_TASK_STATICTICS
      : statInsertNum(0), statGetNum(0)
#endif /* PF_TASK_STATICTICS */
    {}

    /*! All threads can insert a task. We need to lock */
    INLINE void insert(Task &task);
    /*! Only the owner can pick up tasks. No need to lock */
    INLINE Task* get(void);

#if PF_TASK_STATICTICS
    void printStats(void) {
      std::cout << "insertNum " << statInsertNum <<
                   ", getNum " << statGetNum << std::endl();
    }
    Atomic32 statInsertNum, statGetNum;
#endif /* PF_TASK_STATICTICS */
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
    /*! Try to get a task from all the current queues */
    INLINE Task* getTask(int threadID);
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
    enum { queueSize = 2048 };    //!< Number of task per queue
    static THREAD uint32 threadID;//!< ThreadID for each thread
    TaskWorkStealingQueue<queueSize> *wsQueues;//!< 1 queue per thread
    TaskAffinityQueue<queueSize> *afQueues;    //!< 1 queue per thread
    FastRand *random;             //!< 1 random generator per thread
    thread_t *threads;            //!< All threads currently running
    size_t threadNum;             //!< Total number of threads running
    size_t queueNum;              //!< Number of queues (should be threadNum+1)
    volatile bool dead;           //!< The tasking system should quit
  };

  /*! Allocator per thread */
  class ALIGNED(CACHE_LINE) TaskStorage
  {
  public:
    TaskStorage(void) :
#if PF_TASK_STATICTICS
      statNewChunkNum(0), statPushGlobalNum(0), statPopGlobalNum(0),
      statAllocateNum(0), statDeallocateNum(0),
#endif /* PF_TASK_STATICTICS */
      allocator(NULL)
    {
      for (size_t i = 0; i < maxHeap; ++i) {
        this->chunk[i] = NULL;
        this->currSize[i] = 0u;
      }
    }
    ~TaskStorage(void) {
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
#endif /* PF_TASK_STATICTICS */

  private:
    friend class TaskAllocator;
    enum { logChunkSize = 12 };            //!< log2(4KB)
    enum { chunkSize = 1<< logChunkSize }; //!< 4KB when taking memory from std
    enum { maxHeap = 10u };      //!< One heap per size (only power of 2)
    TaskAllocator *allocator;    //!< Handles global heap
    void *chunk[maxHeap];        //!< One heap per size
    uint32_t currSize[maxHeap];  //!< Sum of the free task sizes
    std::vector<void*> toFree;   //!< All chunks allocated (per thread)
    ALIGNED_CLASS
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
    enum { maxHeap = TaskStorage::maxHeap };
    enum { maxSize = 1 << maxHeap };
    TaskStorage *local;  //!< Local heaps (per thread and per size)
    void *global[maxHeap]; //!< Global heap shared by all threads
    MutexActive mutex;     //!< To protect the global heap
    uint32 threadNum;      //!< One thread storage per thread
  };

  ///////////////////////////////////////////////////////////////////////////
  /// Implementation of the internal classes of the tasking system
  ///////////////////////////////////////////////////////////////////////////
  template<int elemNum>
  void TaskWorkStealingQueue<elemNum>::insert(Task &task) {
    const TaskPriority prio = task.getPriority();
    assert(head[prio] - tail[prio] < elemNum);
    this->tasks[prio][this->head[prio] % elemNum] = &task;
    this->head[prio]++;
    IF_TASK_STATISTICS(statInsertNum++);
  }

  template<int elemNum>
  Task* TaskWorkStealingQueue<elemNum>::get(void) {
    if (this->getActiveMask() == 0) return NULL;
    Lock<MutexActive> lock(this->mutex);
    const int mask = this->getActiveMask();
    if (mask == 0) return NULL;
    const TaskPriority prio = TaskPriority(__bsf(mask));
    const int32 index = --this->head[prio];
    Task* task = this->tasks[prio][index % elemNum];
    IF_TASK_STATISTICS(statGetNum++);
    return task;
  }

  template<int elemNum>
  Task* TaskWorkStealingQueue<elemNum>::steal(void) {
    if (this->getActiveMask() == 0) return NULL;
    Lock<MutexActive> lock(this->mutex);
    const int mask = this->getActiveMask();
    if (mask == 0) return NULL;
    const TaskPriority prio = TaskPriority(__bsf(mask));
    const int32 index = this->tail[prio]++;
    Task* stolen = this->tasks[prio][index % elemNum];
    IF_TASK_STATISTICS(statStealNum++);
    return stolen;
  }

  template<int elemNum>
  void TaskAffinityQueue<elemNum>::insert(Task &task) {
    const TaskPriority prio = task.getPriority();
    assert(head[prio] - tail[prio] < elemNum);
    Lock<MutexActive> lock(this->mutex);
    this->tasks[prio][this->head[prio] % elemNum] = &task;
    this->head[prio]++;
    IF_TASK_STATISTICS(statInsertNum++);
  }

  template<int elemNum>
  Task* TaskAffinityQueue<elemNum>::get(void) {
    const int mask = this->getActiveMask();
    if (mask == 0) return NULL;
    const TaskPriority prio = TaskPriority(__bsf(mask));
    const int32 index = this->tail[prio]++;
    Task* task = this->tasks[prio][index % elemNum];
    IF_TASK_STATISTICS(statGetNum++);
    return task;
  }

  TaskAllocator::TaskAllocator(uint32 threadNum_) : threadNum(threadNum_) {
    this->local = NEW_ARRAY(TaskStorage, threadNum);
    for (size_t i = 0; i < threadNum; ++i) this->local[i].allocator = this;
    for (size_t i = 0; i < maxHeap; ++i) this->global[i] = NULL;
  }
  TaskAllocator::~TaskAllocator(void) {
#if PF_TASK_STATICTICS
    for (size_t i = 0; i < threadNum; ++i) this->local[i].printStats();
#endif /* PF_TASK_STATICTICS */
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

  void TaskStorage::newChunk(uint32 chunkID) {
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

  void TaskStorage::pushGlobal(uint32 chunkID) {
    IF_TASK_STATISTICS(statPushGlobalNum++);

    const uint32 elemSize = 1 << chunkID;
    void *list = this->chunk[chunkID];
    void *succ = list, *pred = NULL;
    uintptr_t totalSize = 0;
    while (this->currSize[chunkID] > chunkSize) {
      assert(succ);
      pred = succ;
      succ = *(void**) succ;
      this->currSize[chunkID] -= elemSize;
      totalSize += elemSize;
    }

    // If we pull off some nodes, then we push them back to the global heap
    if (pred) {
      *(void**) pred = NULL;
      this->chunk[chunkID] = succ;
      Lock<MutexActive> lock(allocator->mutex);
      ((void**) list)[1] = allocator->global[chunkID];
      ((uintptr_t *) list)[2] = totalSize;
      allocator->global[chunkID] = list;
    }
  }

  void TaskStorage::popGlobal(uint32 chunkID) {
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
    this->currSize[chunkID] = ((uintptr_t *) list)[2];
  }

  void* TaskStorage::allocate(size_t sz) {
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

  void TaskStorage::deallocate(void *ptr) {
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
    wsQueues(NULL), afQueues(NULL), threads(NULL), dead(false)
  {
    if (threadNum_ < 0) threadNum_ = getNumberOfLogicalThreads() - 2;
    this->threadNum = threadNum_;

    // We have a work queue for the main thread too
    this->queueNum = threadNum+1;
    this->wsQueues = NEW_ARRAY(TaskWorkStealingQueue<queueSize>, queueNum);
    this->afQueues = NEW_ARRAY(TaskAffinityQueue<queueSize>, queueNum);

    // Also one random generator for *every* thread
    this->random = NEW_ARRAY(FastRand, queueNum);

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
    // the scheduler has a reference on the task now
    task.refInc();
    const uint16 affinity = task.getAffinity();
    if (affinity > this->queueNum)
      wsQueues[this->threadID].insert(task);
    else
      afQueues[affinity].insert(task);
  }

  TaskScheduler::~TaskScheduler(void) {
    if (threads)
      for (size_t i = 0; i < threadNum; ++i)
        join(threads[i]);
    SAFE_DELETE_ARRAY(threads);
#if PF_TASK_STATICTICS
    for (size_t i = 0; i < queueNum; ++i) {
      std::cout << "Task Queue " << i << " ";
      wsQueues[i].printStats();
    }
#endif /* PF_TASK_STATICTICS */
    SAFE_DELETE_ARRAY(wsQueues);
    SAFE_DELETE_ARRAY(afQueues);
    SAFE_DELETE_ARRAY(random);
  }

  THREAD uint32 TaskScheduler::threadID = 0;

  Task* TaskScheduler::getTask(int threadID) {
    // Task with affinities have the priority
    Task *task = this->afQueues[threadID].get();
    if (task)
      return task;
    // Then, our own tasks
    else if ((task = this->wsQueues[threadID].get()) != NULL)
      return task;
    // Then, we try to steal some task from another thread
    else {
      const unsigned long index = this->random[threadID].rand() % queueNum;
      return this->wsQueues[index].steal();
    }
  }

  void TaskScheduler::threadFunction(TaskScheduler::Thread *thread)
  {
    threadID = thread->tid;
    TaskScheduler *This = &thread->scheduler;

    // We try to pick up a task from our queue and then we try to steal a task
    // from other queues
    for (;;) {
      Task *task = NULL;
      for (;;) {
        task = This->getTask(threadID);
        if (task) break;
        if (UNLIKELY(This->dead)) goto end;
      }

      // Execute the function
      task->run();
      Task *toRelease = task;

      // Explore the completions and runs all continuations if any
      do {
        const atomic_t stillRunning = --task->toEnd;

        // We are done here
        if (stillRunning == 0) {
          // Start the tasks if they become ready
          if (task->toBeStarted) {
            task->toBeStarted->toStart--;
            if (task->toBeStarted->toStart == 0)
              This->schedule(*task->toBeStarted);
          }
          // Traverse all completions to signal we are done
          task = task->toBeEnded.ptr;
        }
        else
          task = NULL;
      } while (task);

      // run function is counterpart of schedule. We remove one ref
      if (toRelease->refDec()) DELETE(toRelease);
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

#if PF_TASK_USE_DEDICATED_ALLOCATOR
  void *Task::operator new(size_t size) { return allocator->allocate(size); }
  void Task::operator delete(void *ptr) { allocator->deallocate(ptr); }
#endif /* PF_TASK_USE_DEDICATED_ALLOCATOR */

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

