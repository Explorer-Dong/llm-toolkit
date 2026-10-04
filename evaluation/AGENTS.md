# Evaluation

## Single-turn benchmark

The code style and file organization within the AIME subfolder represent the "single-turn" evaluation code that best aligns with my development habits. The implementation of this code should serve as a reference when reproducing any single-turn benchmarks.

## Multi-turn benchmark

The basic code style aligns with the Single-turn benchmark; additional requirements include, but are not limited to:

- All network requests must implement a retry mechanism with exponential backoff.
- Function calling must be passed as function arguments rather than enforced through prompt constraints.
