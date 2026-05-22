# Sensory Tools

Because text-based Large Language Models (LLM) based agents cannot "see" the map, it can call the "sensory tools" to help it see the map and determine its next best course of action. The sensory tools ought to be designed so that it matches the real workflow of an air traffic controller as much as possible, to provide context for the primary task of sequencing, deconflicting aircraft arriving at KDFW, and keep runway events overlapping minimal. The agent will have to call the `edit_tools` described in `API_EDIT_TOOLS.md` to preview the effects, read evaluations of the solution by `eval_tools` in `API_EVAL_TOOLS.md` to see how it fares, and commit/save the action, which leads to mutation of the flight trajectories.

Below are some preliminary ideas. Feel free to sharpen them as you see fit.

### Tool 1: 

### Tool 2: