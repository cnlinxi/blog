# cpp并发编程

## 互斥锁和自旋锁

并发编程时，加锁的目的是保证共享资源在任意时间内，只有一个线程可以访问，避免数据共享导致的错乱。最底层是互斥锁和自旋锁，其它高级锁比如读写锁、悲观锁、乐观锁等都是基于它们实现。

假如一个线程加锁成功，其它线程试图加锁则会失败，失败线程的处理方式如下：

- **互斥锁**加锁失败后，线程释放CPU，给其它线程。
- **自旋锁**加锁失败后，线程会忙等待，直到拿到锁。

因此，持有互斥锁的*失败*线程会退出，等待锁释放时自己被系统唤醒；而持有自旋锁的*失败*线程会“while循环”反复争夺资源。但由于互斥锁加锁失败会进行上下文切换，从而引入一定开销，因此如果锁住的代码执行时间比较短，等待开销小于上下文切换的开销，此时互斥锁就有优势。

### 互斥锁

互斥锁是一种“独占锁”，竞争加锁失败的线程会释放掉CPU，自然该线程加锁的代码就会被阻塞。互斥锁加锁失败而阻塞的现象是由操作系统内核实现的，当互斥锁加锁失败，就会从用户态进入内核态，内核会切换线程，此时会有两次线程上下文切换的性能成本：

- 当线程加锁失败时，内核将线程从“运行”状态设置为“睡眠”状态，然后将CPU切换给其它线程使用。
- 当锁释放时，之前“睡眠”状态的线程会变为“就绪”状态，然后内核会在合适的时间将CPU切换给该线程使用。

当两个线程同属于一个进程，在线程上下文切换时，由于虚拟内存是共享的，因此不需要变动，只需要切换线程的私有数据、寄存器等不共享的数据。

在多核环境下，执行`test and set`无法确保操作的原子性，因此互斥锁的原理是对内存总线进行加锁。

### 自旋锁

自旋锁会一直自旋，利用CPU周期，直到锁可用。在单核CPU上，需要抢占式的调度器，即通过时钟终端一个线程，运行其它线程。否则，自旋锁在单CPU上无法使用，因为一个自旋的线程永远不会放弃CPU。

当加锁失败时，互斥锁进行“线程切换”，自旋锁进行“忙等待”。`忙等待`可以用while循环来实现，但最好使用CPU提供的`PAUSE`指令来实现忙等待。

如果被锁住的代码执行时间很短，那么“忙等待”时间相应也很短，此时适合采用自旋锁。但实际使用时，互斥锁更为普遍。但无论使用何种锁，加锁的代码范围应尽可能小，也就是加锁的粒度要尽可能细，以加快执行速度。

## 互斥锁的基本概念

互斥量（mutex）提供了独占所有权的概念，可以控制对资源的访问。信号量（semaphore）则是一个计数器，限制了并发访问同一资源的线程数量。

[Standard C++](https://isocpp.org/wiki/faq/cpp11-library-concurrency)在创建信号量时，计数器的值总是在0和最大值之间，当计数器的值严格大于0时，对`Wait()`的调用会立刻返回，并将计数器的值减一；当计数器的值为0时，对`Wait()`的调用会阻塞。对于阻塞的信号量，只有`Signal()`调用后，计数器的值重新大于0，此时才会返回。信号量适用场景为：同一时刻只有固定数量消费者访问共享资源，比如信号量可以看做酒店中可预订的房间数量，房间被预定表示一次对信号量`Wait()`的调用，退房表示对`Signal()`的调用：

```cpp

#include <mutex>
#include <condition_variable>

// Simplest implementation
class Semaphore {
public:
  explicit Semaphore(int count = 0) : count_(count) {}

  // 释放一个信号量，计数器加一
  void Signal() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++count_;
    cv_.notify_one();
  }

  // 消耗一个信号量，计数器减一
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [=] { return count_ > 0; });
    --count_;
  }

private:
  std::mutex mutex_;  // 互斥量，表示独占所有权
  std::condition_variable cv_;  // 条件变量
  int count_;  // 信号量的计数器
};
```

## mutex

### 头文件

cpp中mutex和锁类型的类和函数都声明在`<mutex>`头文件中。

#### mutex类（四种）

- `std::mutex`：最基本的mutex类。
- `std::recursive_mutex`：递归mutex类，允许同一个线程对互斥量多次上锁（也即递归上锁），释放互斥量时也需要调用和加锁次数相同的`unlock()`。
- `std::time_mutex`：定时mutex类，成员函数`try_lock_for()`接受一个时间范围，如果在该时间范围内线程没有加锁成功则阻塞，`try_lock_until()`接受一个时间点，如果在该时间点之前没有加锁成功则阻塞。
- `std::recursive_timed_mutex`：定时递归mutex类。

#### lock类（两种）

- `std::lock_guard`：提供线程对互斥量自动加解锁的功能。
- `std::unique_lock`：提供线程对互斥量自动加解锁的功能，并可以中途解锁。可配合条件变量`condition_variable`解决“生产者-消费者”问题。

#### 函数

- `std::lock`：对互斥量加锁，加锁失败则本线程阻塞。
- `std::try_lock`：尝试对互斥量加锁，加锁失败本线程不会阻塞。
- `std::call_once`：多个线程同时调用某个函数，`call_once()`可以确保多个线程只调用该函数一次。

### 构造函数

`std::mutex()`不允许拷贝构造，也不允许move拷贝，最初产生的mutex对象处于解锁（unlock）状态。

### 示例

```cpp

#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>          // std::mutex

volatile int counter(0); // non-atomic counter
std::mutex mtx;          // 定义互斥量，该互斥量控制非原子计数器counter的自增

void attempt_10k_increases() {
    for (int i=0; i<10000; ++i) {
        if (mtx.try_lock()) {  // 尝试加锁，也即仅在未加锁状态下才会自增counter
            ++counter;
            mtx.unlock();  // 解锁
        }
    }
}

int main (int argc, const char* argv[]) {
    std::thread threads[10];
    for (int i=0; i<10; ++i)
        threads[i] = std::thread(attempt_10k_increases);

    for (auto& th : threads) th.join();
    std::cout << counter << " successful increases of the counter.\n";

    return 0;
}
```

## lock_guard和unique_lock

为了方便mutex加解锁，避免加锁后忘记解锁，cpp引入`lock_guard`和`unique_lock`实现自动加锁与解锁功能，这有点类似于普通指针和智能指针之间的关系。

### lock_guard

`lock_guard`在构造函数时加锁，在析构函数时解锁。比如：

```cpp

mutable std::mutex mutex_;  // 定义互斥量

bool Empty() const {
std::lock_guard<std::mutex> lock(mutex_);  // 调用lock_guard构造函数，此时加锁
return queue_.empty();
}  // 大括号{}结束，离开作用域，调用lock_guard析构函数，自动解锁
```

在实例化`lock_guard`对象时会调用构造函数加锁，在离开作用域时`lock_guard`会被销毁，自动解锁，但如果这个作用域比较大，加锁的代码范围会偏大，从而影响执行效率。

### unique_lock

`unique_lock`同样会在构造函数时加锁，在析构函数时解锁。但可以利用`unique_lock.unlock()`来解锁，或者可以配合“条件变量”（condition variable）等使用，在析构时会判断当前锁的状态以决定是否解锁，因此可以方便地控制锁的粒度。而`lock_guard`在析构时一定会解锁，也没有中途解锁的功能。`unique_lock`内部会维护一个锁的状态，所以效率会比`lock_guard`慢。

## condition_variable

条件变量（condition variable）的一般用法是：线程A等待某个条件并挂起，直到线程B设置并通知条件变量，线程A才会被唤醒。条件变量可解决经典的“生产者-消费者”问题。

等待的线程可能有多个，因此通知线程可以选择一次通知一个`condition_variable.notify_one()`，还是一次通知所有等待线程`condition_variable.notify_all()`。比如：

```cpp

mutable std::mutex mutex_;  // 定义互斥量

void Push(const T& value) {
{
  std::unique_lock<std::mutex> lock(mutex_);  // 调用构造函数，此时加锁
  // 条件变量被通知后，本线程被唤醒，但有可能是超时等假唤醒，因此需要while检查条件是否满足
  while (queue_.size() >= capacity_) {
    // wait()解锁并将本线程挂起，CPU交给其它线程使用，等待唤醒
	not_full_condition_.wait(lock);
  }
  queue_.push(value);
}  // 离开此大括号，调用析构函数，自动解锁
not_empty_condition_.notify_one();  // 通知一个其它线程
}
```

由于`lock_gurad`不能中途解锁，因此和条件变量（condition variable）搭配使用的锁必须是`unique_lock`，而不能是`lock_guard`。

条件变量被通知后，挂起的线程会被唤醒，但是唤醒有可能是超时等异常情况导致的假唤醒，因此被唤醒的线程需要检查条件是否满足，因此`wait()`要放到条件循环中，确保是“真唤醒”。

> [C++11 并发指南系列 - Haippy - 博客园](https://www.cnblogs.com/haippy/p/3284540.html)
> [C++11 并发指南三(std::mutex 详解) - Haippy - 博客园](https://www.cnblogs.com/haippy/p/3237213.html)
> [C++11 并发指南五(std::condition_variable 详解)](https://www.cnblogs.com/haippy/p/3252041.html)
> [C++11多线程编程(三)——lock_guard和unique_lock - 知乎](https://zhuanlan.zhihu.com/p/340348726)
> [对比介绍：互斥锁 vs 自旋锁 - 知乎](https://zhuanlan.zhihu.com/p/297929103)
> [Mutex and Semaphore - 知乎](https://zhuanlan.zhihu.com/p/87387993)
> [C++ 多线程 (4) 互斥量（mutex）与锁（lock） - 一抹烟霞 - 博客园](https://www.cnblogs.com/long5683/p/12997011.html)