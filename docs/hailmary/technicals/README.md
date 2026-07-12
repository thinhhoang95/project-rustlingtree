# Hailmary technical documentation

This directory explains the implemented `hailmary` module from a maintainer's
point of view. It is intentionally separate from the original design document:
the design records the problem and intended milestone, while these documents
describe the code that now exists.

Start with [Architecture](architecture.md). It contains the complete mental
model, important coordinate and time conventions, major contracts, and the
full path from source data to evaluated rollouts.

Use the other documents when you need more detail:

- [End-to-end data flow](end-to-end-data-flow.md) follows each input, artifact,
  event, action, feature, and outcome through the system.
- [Component reference](component-reference.md) maps every `hailmary`
  subpackage to its responsibilities, public concepts, dependencies, and
  consumers.

The implementation is under `src/hailmary`. The executable validation examples
are under `notebooks/hailmary`, and focused tests are under `tests/hailmary`.

