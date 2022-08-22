# Template Task Graph User Guide {#userguide}

## Contents {#toc}

* [Building and Installing TTG](https://github.com/TESSEorg/ttg/blob/master/INSTALL.md)
* [Your First TTG Program](#firstprog)
* [Compiling Your First TTG Program](#compiling)
* [Data-Dependent Program](#datadependent)
* [Streaming Terminals](#streamingterminals)
* [Distributed Computing](#distributedcomputing)

## Your First TTG Program {#firstprog}

The following code creates four tasks, one of type A, two of type B
(B(0) and B(1)), and one of type C, and ensures that A runs before
both B tasks, and that both B tasks run before C.

\dotfile simple.dot "Simple diamond DAG"

\dontinclude simple.cc
\skip #include
\until #include

To implement a TTG program, the user just needs to include `ttg.h`. The selection
of the task backend is usually done at compile time through a compiler definition.
However, before include `ttg.h`, the user could `#define TTG_USE_PARSEC 1` or
`#define TTG_USE_MADNESS 1`. Note that only one of the backends can be defined,
and the recommended way is to define which backend is used when invoking the
compiler (see [Compiling Your First TTG Program](#compiling) below).

\skip static
\until }

This is the function that implements tasks of type A. Tasks of this type have no key
to identify them, and receive no input data. Their only output terminal is defined to
receive a key of type `int` and a value of type `double`.

We use `ttg::print` to printout information, as a convenience function that also
avoids messages from multiple threads to interfere with each other.

The task sends the value `1.0` to `B(0)` by outputing the key of type `int` and
value `0` with the data of type `double` and value `1.0` on the output terminal of
index `<0>`, and the key of type `int` and value `1` with the data of type `double`
and value `2.0` on the same output terminal.

Because the two keys are different, these two `ttg::send` instantiate two different
target tasks. Which task depends on how the function is wrapped into a template
task (TT), and how the terminals of this template task are connected to other 
terminals of template tasks (see below).

\skip static
\until }

This function defines the behavior of tasks of type B.

This time, tasks of type B have an integer identifier (`key`), an input value
(`value`), of type `double`, and two output terminals. Both output terminals have no
identifier (keys of type `void`), and carry a data of type `double`.

The task sends to different terminals depending on the value of `key`: tasks with
a `key` of `0` output on the terminal of index `<0>`, while tasks with another `key`
output on the terminal of index `<1>`. They also output different values on these
edges.

Because the output terminals do not define a task identifier (their keys are of type
`void`), one cannot use `ttg::send`, but needs to use `ttg::sendv`. `ttg::sendv` 
differs from `ttg::send` only in the fact that `ttg::send` requires a key identifier 
for the destination task, while `ttg::sendv` does not.

\skip static
\until }

Tasks of type C are implemented with this function. It's a sink task: no
`ttg::send` are emitted by this task that takes no task identifier, defines no output
terminals, and only an input value of type `double`.

\skip main
\until ttg::initialize

The code needs to initialize ttg before any other ttg-related calls.

\skip Edge
\until B_C1

We define 3 edges, to connect the different tasks together.

They have different prototypes: `A_B` carries an identifier of type `int` and a
value of type `double`, while `B_C0` and `B_C1` carry no identifier (`void`) and
a value of type `double`.

To help debugging, we give unique meaningful names to these edges in the
constructor argument.

We need only three edges, because these edges define connections between
the template tasks, not connections between tasks. Their instantiation by
`ttg::send` or `ttg::sendv` define the actual edges between tasks.

There are two edges connecting `B` to `C` because `C` has two input terminals,
and if we used the same edge between `B` and `C`, sending on that edge would trigger
`C` twice.

\skip wa
\until wc

We now define the three template tasks `wa`, `wb`, and `wc`, using the `ttg::make_tt`
helper.

`ttg::make_tt` takes as parameters the function that implements the task, the list of
input edges that are connected to its input terminals, the list of output edges
that are connected to its output terminals, the name of the task, the list of names
for the input terminals, and the list of names for the output terminals.

These TTs and the edges define the template task graph that will then be
instantiated as a DAG of tasks by the execution.

\skip make_graph_executable
\until invoke

Before executing the first tasks, the template task graph must be made executable
by calling `ttg::make_graph_executable()` on each source TT of the graph. This signals to the runtime system that all edges that connect TTs are defined, computes
internal state necessary to track all dependencies, and registers active message
handles for each template task type.

We need to start the DAG of tasks by invoking the initial tasks with `ttg::TTBase::invoke()`.
In this simple  DAG, there is a single initial task, the task A, which runs on rank 0.

\skip execute
\until fence

We can then start the execution of the DAG of tasks. This will enable the 
compute-threads in the ttg library, and start instantiating tasks as the execution 
unfolds.

With `ttg::fence()`, we wait for the completion of all DAGs started.

\skip finalize
\until }

And finally, we can shut down the ttg library and return from the application.

\ref simple.cc "Full first example"

## Compiling Your First TTG Program {#compiling}

The recommended way to compile a TTG program is to use CMake.

Below, you will find a minimal CMakeLists.txt file to compile the first 
example above with both the PaRSEC and the MADNESS driver.

~~~~~~~~~~~~~{.cmake}
cmake_minimum_required(VERSION 3.19)
project(TTG-Example CXX)

find_package(ttg REQUIRED)

add_executable(first-parsec first.cc)
target_compile_definitions(first-parsec PRIVATE TTG_USE_PARSEC=1)
target_link_libraries(first-parsec PRIVATE ttg-parsec)

add_executable(first-mad first.cc)
target_compile_definitions(first-mad PRIVATE TTG_USE_MADNESS=1)
target_link_libraries(first-mad PRIVATE ttg-mad)
~~~~~~~~~~~~~

This CMakeLists.txt uses `find_package(ttg)` to define the different ttg targets.
`find_package` uses the `ttg_DIR` CMake variable as a hint where to find
configuration files. So, if you installed ttg in `/path/to/ttg`, you can point
`find_package` to the appropriate directory by calling CMake as follows:

~~~~~~~~~~~~~{.sh}
cd /path/to/your/builddir
cmake -Dttg_DIR=/path/to/ttg/lib/cmake/ttg /path/to/your/sourcedir
~~~~~~~~~~~~~

`find_package(ttg)` defines the following CMake targets:
  - `ttg-parsec`: the PaRSEC backend for TTG
  - `ttg-mad`: the MADNESS backend for TTG

When source code `#include <ttg.h>`, it needs to define which backend it uses.
In this example, we do that from the command line, by adding the compile-definition
`TTG_USE_PARSEC=1` or `TTG_USE_MADNESS=1`. 

It is then sufficient to tell CMake that the executable depends on the
corresponding TTG target to add the appropriate include path and link
commands.

## Data Dependent Program {#datadependent}

We now extend the first example to illustrate a data-dependent application
behavior. Consider now that the tasks of type C can dynamically decide to
iterate over the simple DAG of tasks before, depending on the values
received as input.

To make the example simple, we will simply define a threshold: if the data
sent by B(0) plus the data sent by B(1) is lower than this threshold, then
the DAG should be iterated, otherwise the application is completed.

One way of representing this behavior is denoted by the graph below:

\dotfile iterative.dot "Iterative diamond DAG"

First, because each task in the DAG needs to be uniquely identified, and there
are potentially many tasks of type A or C, tasks of these kinds now need to 
get an identifier. Second, tasks of type B are not only identified by 0 or 1, but
also need another identifier that denotes to which task of A or C it is
connected. We extend the identifier type of B to `Key2`, which is a `std::pair<int, int>`
to do this simply.

Second, the function that implements the task for C needs to decide dynamically
if it continues iterating or not. This is done by conditionally calling `ttg::send`
in this function. If the function does not call `ttg::send`, then the no more
task is discovered, and the whole operation will complete.

\dontinclude iterative.cc
\skip #include
\until #include <ttg.h>

The inclusion of the `ttg/serialization/std/pair.h` file is necessary to import
the serialization mechanisms for the task identifiers of tasks of type A or C. 

\skip const
\until const

We define the threshold as a globally visible constant.

\skip using
\until // namespace std

We define the key type as a `std::pair<int, int>`, and extend the `std::operator<<` 
to printout an object of type `Key2`

\skip static
\until }
\until }
\until }

Tasks of type A now take an integer key, and an input value; the output is modified
to take a `Key2` (as tasks of type B have keys of type `Key2`).

\skip static
\until }

Tasks of type B now take a key of type `Key2`, and the output is modified
to take an integer key. We then use `ttg::send` instead of `ttg::sendv`, 
because `ttg::sendv` is only used to send to a task that does not have a key identifier.

\skip static
\until }
\until }
\until }

Tasks of type C are modified the same way, and the function that implements the
task holds the dynamic decision to continue in the DAG or not.

\skip main
\until C_A

We update the edges types to reflect the new tasks prototypes, and add a new
edge, that loops from C(k) to A(k+1) (note that the value of the key is decided
in the function itself, this has no impact on this part of the code).

\skip wa
\until wc

The `ttg::make_tt` calls are also updated to reflect the new task prototypes, and include
the edge from C(k) to A(k+1).

\skip make_graph_executable
\until invoke

When invoking A(0, 0.0), one needs to provide the key for the task and the
input value for each input that A now defines.

\skip execute
\until }

\ref iterative.cc "Full iterative diamond example"

## Streaming Terminals {#streamingterminals}

Now, consider that for a given k, there can be a large amount of tasks
of type B, and that the number of such tasks depends on some computation.
This means that the input of tasks of type C is not fixed, but variable.

To express such construct, it is possible to do it by building a sub-DAG
of tasks that combine the outputs of the different tasks of class B before
passing the combination to task C.

TTG provides a more synthetic construct to do so easily: the streaming
terminals.

\dotfile reducing.dot "DAG of the iterative diamond of arbitary width"

The begining of the program remains identical to the iterative
case above: we still use a `std::pair<int, int>` that we alias as
`Key2` to define the task identifiers of tasks of class B, and we
use the standard serialization provided by TTG for those.

\dontinclude reducing.cc
\skip #include
\until // namespace std

The code for tasks of type A will be inlined as a lambda function,
because it needs to access other parts of the DAG that need to be
defined before. The code for tasks of type B becomes simpler: 
we always send the updated input that tasks of type B receive
to the single input terminal of tasks of task C, so we don't need to
differentiate between the keys to decide on which output terminal
to provide the data.

\skip static
\until }

Tasks of type C have been simplified too: they now take a single
input, and it's the input terminal that will do the sum operation.

\skip static
\until }
\until }
\until }

The main program that builds the DAG starts similarly to the
simple iterative diamond example. Edge types have been simplified,
because there is less unique edges (but edges of type `C_A` will
be extended to include the streaming capability).

\skip main
\until C_A

Tasks of type C are defined first, because we need to expose those
to the code of tasks of type A.

\skip auto
\until wc

Now, we define the input reducer function to apply to the input
terminal 0 of tasks of type C. The `set_input_reducer` function
takes two references to elements of the appropriate type, `a` and `b`.
The operation goal is to aggregate the values as they are sent
to the input terminal. The first time a data is sent to this
input terminal, it is copied onto the current aggregated value.
Every other data sent to the same input terminal (and for the same
destination task) is reduced into the aggregator value via this
lambda. `a` is a reference to the (mutable) aggregator value, while
`b` is a reference to the (constant) value to add.

Here, the function we define simply adds the value of `b` to `a`.

\skip set_input_reducer
\until set_input_reducer

We can now define the tasks of type A. Instead of passing the
function to call, we define it in a lambda expression, which allows
us to capture the TT of type C (`wc`). The prototype of this lambda
is the one expected for tasks of the A. After displaying its name,
the task calls `set_argstream_size` on the first input (`<0>`) of `wc`.
This function takes two arguments: a task identifier (`k`), and the
number of elements that are expected as input of the streaming terminal
`<0>`. That counter can be data depndent, in this case we set it to
`k+1`.

\skip wa
\until set_argstream_size

The task can then create as many tasks of type B as is needed, 
and since each task of type B will output their value into the
streaming terminal of the corresponding C, we instantiate `k+1`
tasks of type B by sending them input data.

The other parameters are the usual parameters of `ttg::make_tt`.

\skip for
\until ttg::edges

Tasks of type B are created according to the new prototype,
and the rest of the code is unchanged.

\skip auto
\until EXIT_SUCCESS
\until }

\ref reducing.cc "Full iterative diamond of arbitrary width example"

## Distributed Computing {#distributedcomputing}

Any TTG program is a parallel application. In the current backends,
TTG applications are also MPI applications. Tasks are distributed
between the MPI ranks following a process keymap. The default
process keymap hashes the task identifiers and distributes the
hashes in a round-robin way. The user can control the task distribution
by setting a user-defined keymap for Task Templates.

In the iterative diamond of arbitrary width, we can easily provide
a suitable keymap by pinning tasks of type A and C (which are the
first and last task of each diamond) onto the rank 0, while distributing
the tasks of type B between the ranks using the second element in
the key of those tasks.

This gives the code below, almost identical to the previous example,
except for the keymap definition, and displaying on which rank each
task executes.

\include distributed.cc

\ref distributed.cc "Full iterative diamond of arbitrary width example with user-defined keymap"
