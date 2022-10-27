# Device Task Design {#Design-Device}

### problem statement
TTG must be able to execute general user-defined graphs on machines with heterogeneous execution and address spaces, e.g., using multiple processes each having multiple CPU threads + device streams, with each thread/stream preferring or limited to a specific address range.

### key concerns
The key issues are how to manage:
- the asynchrony of the device programming models, and
- the heterogeneity of the address space.

There are multiple "solutions" to each issue, hence there are many possible designs. I'll discuss each issue first, then outline the aggregate designs we are pursuing.

#### Memory:
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

#### Asynchrony
- *Stages*: decompose tasks into stages, with runtime-managed periods between stages for managing asynchrony of the actions scheduled by each stage
    - pro: most explicit, easier to reason about
    - con: most verbose; device-capable tasks look very different from host tasks
- *"Threads"*: use conventional mechanisms (threads for current backends, coroutines/fibers usable).
    - pro: least host/device dichotomy
        - threads: tasks are ordinary functions
        - fibers/coroutines: task functions "look" like ordinary functions (and can be made almost like normal functions using macros)
    - con: potentially lower performance
        - threads: due to the need to context switch to "yield" to other tasks
        - fibers/coroutines: implementation details are more complex and usually involve heap allocation (C++20 coro)
    - con: runtime is still responsible for managing the executor space heterogeneity (control where to launch a task) and asynchrony (events/host callbacks).

#### other considerations

- it's not possible to manage memory from the device code, hence all program logic, including _device-capable_ tasks, must execute on host executors. In principle if we restricted ourselves to a single-source language (SYLC-extended C++) we could write device capable tasks directly as device code, but current language limitations mandate wrapping everything into host code.

#### Current designs
- *UM+threads*: use UM for memory management + threads for asynchrony
- *NM+stages*: use Parsec's native memory pool manager + stage-decomposed tasks
