**Code style — orthodox, ultra-transparent**

**1. Types and values**

Spell the type. Use descriptive shorthand. Keep types visible at their declarations; put generic machinery behind named interfaces.

Distinguish types whose confusion causes bugs: byte offsets and character positions, identities and indices, samples and seconds. Specify conversion rules, including rounding and encoding. Wrap scalars when the distinction matters.

Choose numerical precision once. Use doubles throughout a computation specified in doubles. Keep storage reusable without adding precision, layout, execution, and allocator parameters to every algorithm.

Name fields so callers need not memorize positions. Group related data in records. Use enums for alternatives and named factories when several fields must agree. Initialize independent options directly. Initialize each variable explicitly.

State array shape, active length, axes, and layout. Distinguish capacity from live length and maximum bounds from active regions. Fixed dimensions represent real bounds. Choose index types that support the required arithmetic; account for negative differences and signed/unsigned conversions.

**2. Statements and expressions**

Write in execution order. Use assignments, conditionals, counted loops, early exits, and named calls. Show traversal when indices, order, boundaries, or writes explain the algorithm.

Name executable behavior. No lambdas, implicit captures, anonymous callbacks, or comprehensions. Use named functions, methods, functors, or listeners. Give retained state named fields so its dependencies and lifetime can be inspected.

Keep each expression to one recognizable operation. Keep formulas intact; name useful intermediate quantities. Avoid tiny arithmetic wrappers and expressions that combine allocation, conversion, decisions, and writes.

Calculate, then return. Return a named value, literal status, or direct field or element reference. Keep side effects in separate statements so their order is visible.

Use operators with familiar meanings: complex arithmetic, indexing, value comparisons. Do not hide scheduling, ownership changes, or an allocating pipeline behind punctuation. Whitespace does not change an operator's meaning. Make meaningful conversions explicit.

Break chains that hide intermediate types, evaluation, retained state, or allocation. Name the intermediate value or extract a procedure. State the procedure's effects; a name alone does not explain them.

**3. Repeated kernels**

Choose mode, format, direction, and implementation before the element loop. Hoist conditions only when their values and effects stay invariant throughout the call. Run a named kernel for each distinct operation. Specialize useful cases without generating every combination of unrelated options.

Handle empty and short inputs first. Calculate the interior bounds, then process edges and interior separately. A convolution's interior needs no edge checks. Split contiguous regions where their treatment changes; preserve ordering and overlap semantics.

Validate whole-buffer shape, extent, capacity, format, access, alignment, and non-overlap before the kernel where required. Pass the established count and parameters. Keep checks whose result can change during execution.

Keep loop bounds and storage stable. Use direct indexing; traverse the contiguous dimension innermost where dependencies allow. Avoid per-element allocation, growth, format discovery, implementation selection, and unnecessary pointer chasing. Use named math operations and small helpers with visible effects.

Read fixed parameters once. Accumulate locally and publish afterward when intermediate updates need not be observed. Repeated writes through members or output references introduce possible interactions with input accesses.

Distinguish in-place, disjoint-output, and overlapping-range operations. `const` does not guarantee non-aliasing. Confine compiler-specific restricted-pointer declarations to kernels whose callers satisfy their access rules. Guarantee alignment at the actual access offset; aligned storage does not make every slice equally aligned. Optimization assumptions must describe established facts.

Keep element-dependent branches, especially guards against invalid access or unnecessary work. Do not replace them with arithmetic that evaluates both alternatives or changes NaN and infinity behavior. A ternary does not guarantee branchless instructions. Avoid extra passes and temporary masks solely to remove an `if`.

Specify permitted numerical changes. Reassociation, partial accumulators, fused operations, reciprocal substitution, and assumptions about NaNs or signed zero can change results. Keep them within the required error and exceptional-value behavior. Do not enable blanket fast-math without granting those freedoms.

Keep allocation, logging, locking, callbacks, and error construction outside per-element arithmetic unless those effects are the operation. Inlining, unrolling, and vectorization hints do not resolve dependencies. Force expansion only for a specific benefit; duplicated code and excess live temporaries can make execution worse.

**4. Procedures and data**

Identify inputs, outputs, and permitted mutation. Use `in`, `out`, and `in_out` where available; otherwise use types, names, and a short contract. Mark read-only inputs.

Fill supplied storage for bulk transformations. Return scalars directly when clearer. State required destination size, allocation, retained arguments, and permitted overlap so callers can compose operations safely.

Name intermediate forms: windowed samples, spectrum, threshold, mask, reconstructed output. Show the stages between them. Retain repetition that exposes distinct stages; extract repeated work with the same contract.

In-place operations may use scratch storage. Preserve any input still needed before overwriting it. Require a separate destination where the operation depends on that separation.

**5. Memory use**

Give data an owner and a lifetime. Use ordinary objects, dot access, indexing, and typed references whether storage is local or on the heap. Allocation changes storage location; it need not change the computation's vocabulary.

Keep small temporaries local. Put persistent state and reusable workspaces on their owner. Allocate substantial or runtime-sized storage before repeated processing, then reuse it. Shared scratch fields create dependencies between calls; use them deliberately.

Tie cleanup to the owner's lifetime. Transfer ownership explicitly and leave the former owner empty. Use shared ownership for genuinely shared lifetimes; prevent cycles and keep observational links non-owning. Borrowing grants access without ownership.

Pass read-only references for observation and mutable references for modification or output. Work directly: `processor.process(input, output)`, `frame.samples[i]`, `image.clear()`. Bind an owning handle to a typed reference for repeated access within its valid lifetime.

Keep raw addresses at allocation, borrowing, foreign-call, device, and low-level kernel boundaries. State the extent and lifetime of borrowed memory. Keep size, capacity, access, and release controls on the object so callers need not coordinate them separately.

Use contiguous, length-bearing arrays. Index live elements directly; use dot-accessed controls for size, filling, clearing, reservation, and release. Indexing never silently grows storage. A fixed shape can live on the heap; runtime sizing need not imply growth during processing.

Use standard growable collections for growth and ordinary strings for text. Use specialized heap arrays for substantial buffers edited in place or for specific alignment and allocation needs.

Name storage operations precisely. Fill edits live elements. Reserve preserves values while increasing capacity. Reset-size replaces contents, even at the same size. Clear ends element lifetimes and states whether capacity remains. Release frees storage. Name large copies; moving an owner transfers storage without copying contents.

State whether inputs and outputs may share storage. Different names and read-only access do not establish disjointness. Keep borrows valid through use: reallocation, replacement, release, and destruction can invalidate them. Removing an element ends its lifetime even if its bytes remain. A moved allocation can retain its address under a new owner. Retaining a borrow across deferred work or mutation requires a lifetime guarantee.

Check size multiplication, shape products, capacity growth, and alignment before use. Match allocation and release. Initialize every region that will be read, including padding; clear only where the next operation needs it. Define empty-state behavior. Publish replacement storage only after initialization succeeds.

If allocation failure is reported solely by a Boolean, keep element construction and relocation from escaping through another failure mechanism. Preserve old contents on failed replacement when promised; preparing a replacement may require both allocations at once.

Synchronize shared access at its boundary. Give concurrent computations separate mutable workspaces unless sharing is intentional. Ordinary access then operates within established ownership and mutation rights.

**6. Objects and authority**

An object holds related state, invariants, and authorized operations. Compose independent capabilities. Use inheritance for a behavioral relationship; define ownership separately.

Let parents own children and reverse links observe without extending lifetime. A detached object needs a surviving owner or independent lifetime. Cross authority boundaries through named operations or capabilities that specify read, modify, retain, and release rights.

Define the fallback when authority disappears. Reject stale identities, stop revoked callbacks, honor cancellation boundaries, or use an independently owned snapshot. A fallback cannot grant access to a dead owner.

Construct valid state and establish ownership. Keep plain configuration records as data. Name substantial setup, external interaction, and recoverable acquisition when callers need control over them. Distinguish invocation, connection, ownership, and disposal.

**7. Admitted abstractions**

Use abstractions that name useful operations and reduce what callers must remember. Keep their mechanisms inspectable.

Use standard minimum, maximum, clamp, copy, search, accumulation, inner product, and visitation when they express the work. Name predicates. Write a loop when traversal details matter. An inner product needs no private vocabulary of arithmetic wrappers.

Use templates and compile-time execution to implement declared choices. Name generic operations and their requirements. Require explicit participation in lifecycle and registration protocols; a coincidentally named member must not select ownership or construction behavior.

Use explicit alternatives, borrowed spans, typed identities, and automatic resource owners. Localize virtual dispatch, reflection, and type erasure to boundaries requiring runtime variation. Avoid universal callable frameworks, private standard libraries, allocator hierarchies, and speculative portability layers.

Choose sorting by count, order, duplicates, stability, move cost, comparator cost, and scratch space. Keep standard sort where it fits; specialize justified cases. Keep house search to named lower-bound, upper-bound, and exact-membership operations over explicit intervals. Use priority queues for required priority ordering and updates.

Precompute stable tables and reusable plans when saved work justifies storage. State normalization and tolerances. Keep quantities, stages, boundaries, and results recognizable from the mathematics.

**8. Callbacks and execution**

Give callbacks named targets and explicit context. Binding specifies invocation; subscription controls connection; ownership controls lifetime. Keep these responsibilities visible.

Use a non-owning delegate when a function and context pointer suffice. Binding need not allocate. Use an owning callable when state must be retained. Account separately for event storage, snapshots, and retained state.

Specify event order, reentrancy, and revocation. Define connection and disconnection during emission. A snapshot can preserve registration order, skip handlers disconnected before their turn, and defer new handlers. Keep the active target alive if it can dispose of itself during invocation.

Name asynchronous work, completion, cancellation, and shutdown. Assign mutable state to an execution context. Share immutable plans; keep mutable workspaces separate unless synchronized. Mark independent parallel work and specify reduction order when results depend on it.

Adopt configuration at a defined processing boundary. Rebuild dependent state when inputs change. Synchronize cross-thread publication; copying fields at cycle start is not synchronization.

**9. Failure and foreign interfaces**

Specify whether failure leaves output unchanged, empty, partial, or invalid. Validate before mutation where old contents must survive. Prepare replacements before committing and account for peak storage.

Use explicit results for expected alternatives. Use exceptions at operation or subsystem failure boundaries that permit them. Keep routine hot-path decisions ordinary control flow. Clean up on every exit and define resource-exhaustion behavior.

Assert internal preconditions; validate external inputs. Disabled assertions cannot reject invalid sizes or stale identities. Return complete success or failure states so callers need not infer validity from unrelated fields.

Foreign interfaces specify types, counts, ownership, destruction, errors, callback context, and identity validity. Keep language-specific layouts and ownership behind the interface. Contain exceptions where they cannot cross. Use thin wrappers to restore native resource conventions.

**10. C++ spelling**

Fix the language mode and permitted subset.

| Form | Rule |
|---|---|
| Types | Spell local, iterator, parameter, and return types. Use descriptive aliases. No `auto`, `decltype(auto)`, or trailing returns. |
| Member access | Use dot. For pointers, write `(*pointer).member`. |
| Anonymous execution | No lambdas, including captureless predicates and trampolines. Name functions, functors, visitors, and listeners. |
| Other exclusions | No structured bindings, coroutines, or ranges/views pipelines in the default subset. |
| Type machinery | Keep `decltype(expression)` in named private traits, outside ordinary declarations and public signatures. |
| Compile-time work | Admit templates, `constexpr`, `consteval`, `if constexpr`, packs, and folds. Require explicit protocol opt-in; member detection must not choose architecture. |
| States and borrows | Admit `std::span`, `std::optional`, and `std::variant`. State lifetime and alternative semantics; name visitors. |
| Ownership | Admit `std::unique_ptr`. Use `std::shared_ptr` and `std::weak_ptr` for shared lifetime and observation. |
| Callables | Admit `std::function` for owning callables. Name targets; account for copies, allocation, and retained state. |
| Type erasure | No general `std::any` architecture. Confine inert-metadata compatibility fields to their existing contract. |
| Collections | Use `std::vector` for growth, standard strings for text, fixed arrays for real bounds, and heap arrays for explicit buffer ownership. |
| Comparisons | Write required comparisons. No defaulted comparisons or defaulted spaceship. Preserve meaningful value semantics; omit unused operator families. |
| Initialization | Use designated initializers for independent options with valid defaults; named factories for semantic result states. |
| Library operations | Admit `std::complex<double>`, ordinary algorithms, `accumulate`, `inner_product`, `has_single_bit`, string tests, membership tests, and named-predicate erasure. Representation casts require valid representation rules. |
| Exceptions and RTTI | Use them at failure, adapter, and dispatch boundaries that need them. |

Mark read-only inputs `const`. Use explicit constructors and conversions where implicit conversion changes meaning; reject unintended narrowing. Delete copying for unique owners and define moved-from state. Mark consequential results `[[nodiscard]]`. Use `noexcept` only when dependencies cannot throw. Use `static_assert` for compile-time requirements.

A stricter generated-code profile uses named listeners, typed identities, deterministic initialization and connection order, immutable fixed tables, and bounded views. It excludes `std::vector` and capture-backed callables; use a named owner for required runtime-sized storage. Keep these restrictions scoped to that profile.

**11. Source layout and comments**

Give substantial types, storage owners, and state machines meaningful source boundaries. Keep related definitions together. Put data and helper contracts before their orchestration where practical. Make the main processing sequence easy to find.

Use descriptive boundary names and short conventional mathematical names locally. Keep formatting consistent. Omit accessors, wrappers, and indirection that add no invariant or useful operation.

Explain non-obvious dimensions, padding, normalization, lifetime, overlap, disposal order, and boundaries. State why a surprising operation exists. Preserve useful mathematical correspondence and remove stale comments. Keep vocabulary steady through the procedure.
