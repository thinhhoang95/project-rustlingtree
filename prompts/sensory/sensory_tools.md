# Sensory Tools

Because text-based Large Language Models (LLM) based agents cannot "see" the map, it can call the "sensory tools" to help it see the map and determine its next best course of action. The sensory tools ought to be designed so that it matches the real workflow of an air traffic controller as much as possible, to provide context for the primary task of sequencing, deconflicting aircraft arriving at KDFW, and keep runway events overlapping minimal. The agent will have to call the `edit_tools` described in `API_EDIT_TOOLS.md` to preview the effects, read evaluations of the solution by `eval_tools` in `API_EVAL_TOOLS.md` to see how it fares, and commit/save the action, which leads to mutation of the flight trajectories.

Below are some preliminary ideas. Feel free to sharpen them as you see fit.

### Sensory Tool: Vector Assist with Time to Gain

The idea of vectoring is that given a desired time to gain of the flight path, it will compute a dense 2D map (i.e., a map from "cells" to actual time to gain value) for many fixes in the area of interest, then interpolate this 2D map and then pick 2 candidates:
- The best fix already existing in the list (called an identified fix).
- The best fix, which is a geographical coordinate (called a free fix). 

If the difference between two fixes are less than some threshold, say 10s, then choose the fix from the list. 

#### Remarks 
There is an important question: given a candidate fix, how do you alter the flight path (or the flight's fix sequence - note that a fix could either be an identified fix or a free fix). We will use the following logic: 

1. For each fix f, we try to find a projected point on the flight path called f_hat, which is defined as the point belonging to the flight path that possesses the shortest distance from to the fix f. 

2. Call the two fixes sandwiching f_hat f_A and f_B (with f_A the aircraft reached earlier than f_B), We will test for two variants:
  a. From the first fix ever in the sequence, until f_A, then to f, then to f_B. We call this variant a "sandwiched dogleg." 
  b. From the first fix ever in the sequence, until the fix before f_A, then to f, then to f_B. We call this variant a "replaced dogleg." The reason is that you can see f_A got replaced by f.

So that means we will have two 2D map, and the best free/identified fix will be taken across these two maps. This way, we can add additional flexibility into the vectored path. But note down whether the mutation involved the sandwiched dogleg or the replaced dogleg. 

#### Constraints
- You can only allow maximum of 2 vectoring attempts per flight.
- Do not allow any vectoring that result in any tight turn of less than 25 degrees (that also means no U-turn either to prevent bizzare vectoring patterns that make no sense to human operators).
- We define an `OperationalSpaceMask` as a polygon that defines an area where vectoring could happen. For flights coming from the North (including North East or North West), which is a cone defined by two rays connecting the identified fixes: TTT - WLLTR and TTT - PRX. For arrivals from the South (South East or South West), by the cone defined by two rays: TTT-BGTOE and TTT-WAITT. I think the classification of the arrival from which direction had already existed somewhere in the codebase already. It's best to organize this as an individual module, and after running the code, hardcode the mask directly. The idea is that you can intersect with this mask to yield vectoring options.

# General Notes
* The flight path is a string of fix sequence, and is managed by the ScenarioManager. It is imperative that the mutation will happen on the diff-mutated pre-compute artifact path, not on the pre-computed artifact itself because the diff-mutated version is what the API will give to the client.
* Edit tools can modify the same flight trajectory at most two times, but only one attempt is allowed for "replaced dogleg."
* Consider organizing the code appropriately, avoid a very long Python file. Also, if features are already existing somewhere in the codebase, consider reusing the code to ensure consistency. 