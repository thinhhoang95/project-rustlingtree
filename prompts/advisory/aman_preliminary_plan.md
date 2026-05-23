# Arrival Manager (AMAN) Advisor

The Arrival Manager (or AMAN) is responsible for allocating arrivals and departures into slots in a first-come-first-serve style (for each runway), and then output the time-to-gain (in minutes) for each flight in order to *clear all runway event overlaps*. 

The basic idea is similar to any slot allocator: you divide the timeline of each runway into slots. Each slot can be of different length, depending on the arrival runway event occupancy time and the departure runway event occupancy time (retrieve these values from runway overlap evaluator code to ensure consistency). Then you go from left (earliest) to right (latest). For any pair of overlapping runway events, you "slide" the latter forward so that they stop to be overlapping. You will get the the delay assigned to each flight.

Note that we will keep the departure flights fixed, so events associated with the departures are kept fixed (they are separately managed by a flow control system), so if there is an overlap between an arrival and a departure on the same runway, you just slide the arrival to after the departure. Don't slide any flight in advance (to earlier time) i.e., causing a negative delay.

The output should contain all advised time-to-gain in minutes for all arrival flights affected.

## Implementation location
In `src/advisors/aman` directory. Keep the code organized, don't write all code into just one giant Python file.

- Note that the results will have to depend on diff-mutated flight list in `ScenarioManager`.
- Consistency between AMAN suggested values and the runway overlapping evaluator needs to be maintained. In other words, if the agent could achieve exactly the minutes-to-gain dictated by the AMAN, the evaluator should return zero overlapping events (scoped to arrivals only).