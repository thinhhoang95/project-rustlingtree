# Advisory Tools 

Advisory Tools are tools that could provide additional context on demand to help the user or agent to plan their course of action. Compared to Evaluators, Advisory Tools will provide answers to usual what-if questions, such as how many more miles are needed to successfully clear the feasibility condition, how many seconds will be added if we stretch the path by x nautical miles (along-track), or how many extra seconds will be gained if a speed intervention is conducted.

## Advisory Tool Output
Two things that need to be output:
- (Along-track) Miles-to-gain (Extra nautical miles). Positive means more nautical miles to be covered.
- Minutes-to-gain (Extra flight time minutes). Positive means the flight time is longer.

## Three Advisory Tools 
### Basic Operations
1. Feasibility Advisor: will return the remaining (along-track) nautical miles that need to be covered through path stretching (vectoring). It is shown that this distance is invariant to where the stretch is placed. I think in the metadata, in the computation of the longitudinal profile of infeasible trajectory had already contained this information, thus the feasibility advisor will simply return this value. Just to be sure that the correct value is returned, given the context that multiple versions could exist in the ScenarioManager, like the precomputed artifact and the applied diff. For the computation time, I think we use the same technique as the Speed control advisor below (direct longitudinal profile computation then subtract the difference).
2. Vectoring advisor: if the along track distance is extended by some value x; obviously the miles-to-gain equal to whatever was input, but please compute the minutes-to-gain in two cases:
  - Because in managed descent, the Top-of-Descent (TOD) in stretched path could be pushed to later, the idea is that the mintutes-to-gain is due to x with pre-TOD cruise speed.
  - For infeasible flight, use the Feasibility Advisor first to cover the "infeasible part" first, then the remaining of x with the pre Top-of-Descent cruise speed. 
3. Speed control advisor: given some along-track station s_m value, and the new prescribed Calibrated Airspeed value, return the Miles-to-gain and the Minutes-to-gain. I had tried to derive an analytical way to solve this but I think the best way is to launch two longitudinal profile calculation attempts, and take differences. It is simpler and computationally quite the same to the best analytical method.