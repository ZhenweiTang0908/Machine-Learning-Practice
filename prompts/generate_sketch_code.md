# Generate code sketch

This project is for practicing ML methods by implementing them myself instead of directly using ML frameworks.

Write a code sketch of the given ML method with blanks for me to fill in.

Read the derivation pdf under the corresponding method directory before generating the sketch.

---

## Goal

The goal is:
- understand the ML algorithm itself
- understand the implementation workflow
- practice practical Go project structure
- learn some Go mechanisms naturally during implementation

Priority:
1. Help me understand the ML algorithm
2. Let me implement the core algorithmic logic myself
3. Avoid wasting time on low-level math implementation
4. Keep the code simple and practical

---

## Important Philosophy

I should implement the core ML logic myself.

However, I should NOT waste time implementing:
- vector operations
- matrix containers
- basic linear algebra operations
- CSV parsing
- plotting
- utility math functions

You SHOULD use math/basic libraries like `gonum` for low-level mathematical operations. But DO NOT use libraries that directly solve the target ML algorithm.

For example:
- Allowed:
  - matrix multiplication
  - vector norm
  - eigendecomposition utilities
  - random number generation
  - plotting
- NOT allowed:
  - PCA()
  - sklearn-like APIs
  - one-call ML solutions
  - libraries that directly produce the final ML result

The core algorithm steps must still be implemented by me.

---

## Sketch Requirement

Do NOT provide complete implementations.

For important functions:
- provide function signatures
- provide concise but explanatory comments
- provide pseudocode
- provide TODO blocks
- leave key algorithmic parts unfinished

---

## Project Structure Requirement

Write the code inside the corresponding ML method directory.

Requirements:
- at least one `main.go`
- non-main logic should be placed under `src/`
- keep the project structure practical and idiomatic

---

## Teaching Requirement

Give enough information to help me implement the code:
- hints at the head of each file
- hints for important functions
- explanation of important math steps
- explanation of useful library functions if needed

If some library functions are recommended:
- explain what they do
- explain why they are useful
- explain how to use them briefly

Comments should be concise but explanatory.

---

## Go Requirement

Try to naturally demonstrate practical Go concepts:
- slices
- interfaces
- package organization
- error handling
- concurrency only if naturally useful

Do NOT over-engineer the project.

If appropriate, introduce one lightweight design pattern naturally.

---

## Output Requirement

Output in this order:

1. Project structure
2. File-by-file explanation
3. Skeleton code
4. `coach_instruction.md`
5. Step-by-step implementation order

---

## Learning Goal

The coding goal is:
- understand the algorithm deeply
- avoid unnecessary low-level implementation work
- write as little boilerplate as possible
- focus on the ML logic itself

---

Now I'm going to learn: 
