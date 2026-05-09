# Scenario Manager

Project Rustlingtree aims to build a gym environment to support the agent to learn how to do optimal sequencing of aircraft to land at Dallas Fort-Worth (KDFW), both from historical data (ADS-B based), and synthetic data. 

The goal is to plan in detail the implementation of a Scenario Manager, which manages the *schedule of departure*, *schedule of arrival* for now. More things will be added in the future, such as exporting each aircraft's plan, plan edit tools, and evaluation and verification functions, but for now, just maintaining the departure schedule, schedule of arrival is sufficient.

# Requirements
Preferably we will need a central class that manages all the Resources, like `data/adsb/catalogs/2026-04-01_landings_and_departures.csv`. Departures should include the departure time plus compressed ADS-B trajectory from `data/adsb/compressed/adsb_compressed_flights.jsonl`. For arrivals, keep the time of arrival at first fix (i.e., handed over to the agent), the original fix sequence, and the SIMAP trajectory from `data/artifacts/simap_arrival_flights.jsonl`. Because we will later intervene to assign runways and change the fix sequence, the arrival time will depend on the final plan for each flight. You don't need to uniformly sample the path, you can only return a set of points (preferably at trajectory or altitude changepoints) and the client will linearly interpolate them later.

The implementation should be at `src/mcp_tools/scenario_manager`. Please avoid writing large, long code files. Break the modules by their functionalities to smaller structure so they are easy to maintain. 

Then, add `fastapi` API (you may need to setup the whole server), add a couple of endpoints so that the resources are not reloaded everytime the API is requested something; one endpoint returns the departure schedule and ADS-B trajectory, another returns arrival schedule and SIMAP trajectory to the client.
