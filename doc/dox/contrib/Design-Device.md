# Device Task Design {#Design-Device}

## problem statement
TTG must be able to execute general user-defined graphs on machines with heterogeneous execution and address spaces, e.g., using multiple processes each having multiple CPU threads + device streams, with each thread/stream preferring or limited to a specific address range.

## key concerns
The key issues are how to manage:
- the asynchrony of the device programming models, and
- the heterogeneity of the address space.

There are multiple "solutions" to each issue, hence there are many possible designs. I'll discuss each issue first, then outline the aggregate designs we are pursuing.

### Memory:
- *Unified Memory (UM)*: where available, use single address space (unified memory visible to both host and device executors; it is also possible to use pinned host memory for device calls)
    - pro: simplifies memory management by removing the capacity limitation
    - con: still needs user cooperation: all compute data must be allocated on UM heap, this impacts the design of
      user data types, e.g. making them allocator aware, etc.
    - con: the user will likely needs to use pooled memory management for efficiency reasons (e.g., TiledArray uses Umpire)
    - con: still necessary to provide hints to the kernel driver managing UM.
    - con: reasoning about UM driver performance is difficult, its details are opaque and platform dependent.
- *Device Memory (DM)*: using "native" device memory.
    - pro: simpler performance model due to greatest amount of control (by runtime) over execution
    - pro: can work with stack-capable data types
    - con: The amount is limited, hence this memory must be explicitly managed (akin to how a cache is managed).

Additional memory-related concerns common to both models:
- only partial state needs to be transferred to/from the device
    - which part of the state will differ from algorithm to algorithm, hence encoding/constructing such representation cannot use constexpr code (such as traits)
    - the need for _explicit_ handling of object's partial state is shared by both models
        - UM: such optimization may seem automatic (only the pages of the data actually used on the device are transfered) but in practice the data must be explicitly prefetched, hence partial state transfers are not automatic; furthermore, the unit of UM transfer is a page (4k or more), which is too coarse for many applications
        - DM: serialization of an entire object (which can leverage standard RDMA-like serialization), transfering partial state requires explicit annotation 
    - hence it makes sense to make representation of object's partial state (`View`) a first-class concept in both models.

### Asynchrony
- *Continuations/stages*: decompose tasks into _continuations_ (stages), with runtime-managed scheduling of continutations for managing the asynchrony of the actions initiated by each continuation
    - pro: most explicit, easier to reason about, fewest performance implications
    - con: most verbose; device-capable tasks look very different from host tasks
    - con: limited composability
        - difficult to support general computation patterns (e.g. generator continuation, etc.,)
- *"Threads"*: use threads to deal with the asynchrony (in principle could use user-space threads = fibers)
    - pro: least host/device dichotomy
        - tasks are ordinary (synchronous) functions
        - fully composable
    - con: performance implications
        - due to the need to context switch to "yield" to other tasks
        - thus even fully synchronous computations will suffer
    - con: asynchrony artifacts still appear
        - asynchronous calls must be in general annotated (to force synchronous execution and/or to provide hints to the thread scheduler)
- *"Coroutines"*: use C++20 coroutines
    - pro: less host/device dichotomy compared to continuations
        - task functions "look" like ordinary functions (and can be made almost like normal functions using macros) but returning a custom return object (containing return status + handle to the coroutine) instead of void
        - fully composable
    - performance implications
        - pro: no impact on synchronous tasks
        - con: coroutine implementation details are complex and usually involve heap allocation
        - pro: custom allocators can be introduced to elide heap allocation (at the cost of limited generality)
    - con: asynchrony artifacts still appear
        - co_await annotate the spots where execution may need to be suspended
    - con: less mature due to the need for C++20
        - GCC (10+), LLVM (8+) support coroutines
        - TTG and all of its dependencies will be impacted by the raised standard requirement

### other considerations

- it's not possible to manage memory from the device code, hence all program logic, including _device-capable_ tasks, must execute on host executors. In principle if we restricted ourselves to a single-source language (SYLC-extended C++) we could write device capable tasks directly as device code, but current language limitations mandate wrapping everything into host code.
- runtime is still responsible for managing the executor space heterogeneity (control where to launch a task) and asynchrony (events/host callbacks).

## Current designs
- *UM+threads*: use UM for memory management + threads for asynchrony
- *DM+stages*: use Parsec's device memory pool manager + stage-decomposed tasks
- *?M+coroutines*: UM/DM for memory + C++20 coroutines for handling the asynchrony

### Example code: threads vs continuations vs coroutines

How should we map the following host task onto the device?
```cpp
make_tt([](auto& key, auto& data1, auto& data2) -> void {
    double data3 = blas::dot(data1.data(), data2.data());
    if (data3 >= 0.)
        send<0>(data1);
    else
        send<0>(data2);
}
```

Ideally the task will receive `data1` and `data2` already transferred to the memory space(s) accessible from the device execution space:
```cpp
make_device_tt([](auto& key, auto& data1, auto& data2) -> void {
    double data3 = blas::device_dot(data1.data(), data2.data());
    if (data3 >= 0.)
        send<0>(data1);
    else
        send<0>(data2);
}
```
But now `data3` lives in the host memory so in general we must manage its transfer from the device. Hence either:
- all intermediate data must be managed explicitly within the task, or
- except for the cases where user types are aware of multiple memory spaces (but this makes the state of such types asynchronous).

Here are the tentative device versions of this task in each of the 3 approaches (the memory details are omitted).

#### Threads
```cpp
make_tt([](auto& key, auto& data1, auto& data2) -> void {
    // stage 1
    ConstView view1(data1);
    ConstView view2(data2);
    double data3;
    View view3(data3, NewView | SyncView_D2H);
    // depending on the memory model may need to wait here for the transfers to complete
    // could build the waits into View ctors, or need an explicit await()
    
    // stage 2
    cublasDdot(view1.device_ptr(), view2.device_ptr(), view3.device_ptr());
    // if called an async function need explicit await() here
    // also: who/how will view3 be synchronized

    if (data3 >= 0.)
        send<0>(data1);
    else
        send<0>(data2);
}
```
N.B. `make_tt`: this is a regular task.

#### Continuations
```cpp
make_device_tt(
  // stage 1
  [](auto& key, auto& data1, auto& data2) {
    ConstView view1(data1);
    ConstView view2(data2);
    double data3;
    View view3(data3, NewView | SyncView_D2H);
    return {view1, view2, view3};
    }, 
  // stage 2
  [](auto& key, auto& views) {
    auto& [view1, view2, view3] = views;
    cublasDdot(view1.device_ptr(), view2.device_ptr(), view3.device_ptr());
  },
  // stage 3
  [](auto& key, auto& views) {
    auto& [view1, view2, view3] = views;
    if (*view3.host_ptr() >= 0.)
        send<0>(data1);
    else
        send<0>(data2);
    }
}
```
N.B. `make_device_tt` vs `make_tt`: this is a special task.

#### Coroutines
```cpp
make_tt([](auto& key, auto& data1, auto& data2) -> ttg::resumable_task {
    // stage 1
    ConstView view1(data1);
    ConstView view2(data2);
    double data3;
    View view3(data3, NewView | SyncView_D2H);
    co_await sync_views(view1, view2, view3);  // creates list of transfers to be fulfilled by the runtime
    
    // stage 2
    cublasDdot(view1.device_ptr(), view2.device_ptr(), view3.device_ptr());
    co_await;  // syncs view3; since transfers and kernels execute in different streams the runtime will sync kernel stream, then launch transfers, then resume here

    if (data3 >= 0.)
        send<0>(data1);
    else
        send<0>(data2);
    co_return;  // processes sends and destroys coroutine
}, ...);
```
N.B. `make_tt` and `ttg::resumable_task`: this is a regular task but with special return type.
