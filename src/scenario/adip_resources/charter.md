Given a set of arrival procedures from JPEG format, the goal is to extract the machine-readable format for each procedure by processing the charts. For example you will need to output something like:
```yaml
links:
    GRUCH,ALIGN
    ALIGN,UDALL
    UDALL,JAYCI
altitude_constraints:
    GRUCH-ALIGN: 1000,4000
    ALIGN-UDALL: 1000,3000
    UDALL-JAYCI: 1000,3000
    JAYCI: 1000,2300
holdings:
    BIRLE: 035
    GAATZ: 064
    GLOVE: 064
```

# Remarks
- GRUCH-ALIGN is a link between two waypoints: GRUCH and ALIGN. Note that the "lightning-bolt" symbol usually indicates the waypoint name, which usually consists of 5 characters. However, not only waypoints are admitted in the response, but VOR too (see below).
- Beware that the link is directional: to describe the procedure, follow the filled black arrow along the link; ignore the direction indicated by open/white arrowheads.
- Many links (not all of them) will have the radial (magnetic bearing) denoted (like R-038, R-065...). You don't need to do anything about these values.
- Sometimes you might encounter a VOR symbol (like a hexagon enclosed in a square), in that case just use the full VOR name directly like `EL DORADO`. You should not care about the frequency.
- The links could be between VOR and a waypoint, or even a VOR and VOR, not just waypoint vs waypoint only.
- The holdings are the loop-like, usually associated with a waypoint. On the loop, the bearing is usually marked, like 035 for 035 degrees.
- Some altitude values are given in Flight Level like FL180, some are given in plain feet, like 7000. Always convert to plain feet, avoid mixing between FL and feet.
- For altitude constraints, sometimes you might see something like 10000 and *3100. That's the altitude window: from 3100ft to 10000ft. That correspond to two values in `altitude_constraints`.
- The altitude constraints could be for both links and constrait at a waypoint. For example: `UDALL-JAYCI: 1000,3000` means the link from `UDALL` to `JAYCI`, `JAYCI: 1000,2300` means at `JAYCI` only. 
- Sometimes there are just arrows pointing out from a waypoint or a VOR, with a bearing, **but without the destination waypoint or VOR**. Ignore these. These are direct options which we will not model for now.

# Additional Instructions
- You must not add other field besides what's given in the examples. If some values are not available, put 'N/A' instead.
- Check your results carefully before outputting the one final answer.
- Don't add any prose. In the final response, directly begin your response with ```yaml