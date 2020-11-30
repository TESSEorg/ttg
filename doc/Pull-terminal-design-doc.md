# Pull Terminals - Design Notes

### Motivation

- This requirement cropped up from the wavefront example borrowed from Cpp-taskflow.
- In TTG, every operation with different number of input dependencies is created as a separate Op. 
- However, some applications require data from various sources purely for computation, and do not necessarily imply a data dependency for execution of the task.
- Current model of TTG only allows for data only to be PUSHed to subsequent tasks via terminals. In order to push data from different sources for computation, a 'reader' task is required which when run, instantiates all the tasks it can push data to, which is unnecessary. Tasks should be created when data dependencies are satisfied, and not when data is pushed from a source.
- If data can be PULLed by a task when necessary, we can defer running the "reader" tasks until needed.

### Prototype Implementation

- Create an Op for every pull task. Use the Edges to mark terminals as pull terminals. 
- A pull Op can contain 0 or 1 pull terminals as input.
- Key of the puller is sent as input to the pull task to invoke the task. 
- If the pull task itself has a pull terminal as input, this would walk through the DAG in a reverse manner until necessary data is pulled.
- Invoking a pull task is done via a callback.
- A pull task can be invoked at task creation time (eager) or when the task is ready to run with all dependencies satisfied (lazy).

### Pros

- Ensures unique task IDs for every pull task.

### Cons

- No way to map a single datasource / key to multiple keys.
- Helps with delaying tasks, however not very flexible in design.

## Design Recommendations

### Use a map for pull request

- Allows specifying a datasource different from the key.
- Task keys will not be unique and currently there is no way to handle it.

### Callback Model

- Implement a callback for every pull terminal which takes the request (Ex. set of keys) and returns the data.
- Data may be remote, however send currently uses a callback model and can probably be used for pull requests as well.

## Use Cases

- Wavefront Computation
- Generator/Reader tasks
- Adding 3 vectors - one vector is local and need to pull two other vectors.
- SUMMA - pull data multiple times.

## Questions

- Should Pull Op be able to send data to multiple successors? Use cases?
- Cholesky - why pull ops are needed?




