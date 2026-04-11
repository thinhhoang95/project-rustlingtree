This is the recipe to implement a flight simulator sufficient for ATM research with **plausible vertical rate, airspeed, and groundspeed**, and still makes deceleration **feasible and wind-sensitive**. It uses OpenAP only as a **parameter backend**, not as the simulator core. OpenAP already exposes the pieces you need: aircraft and engine properties, ISA atmosphere and airspeed conversions, WRAP kinematic parameters for descent/final approach/landing, and separate drag/thrust modules. The `prop` package exposes fields like wing area, drag coefficients, OEW/MTOW/MLW/VMO/MMO, and engine metadata; `aero` provides ISA atmosphere and CAS/TAS/EAS/Mach conversions in SI units; WRAP exposes parameters such as `descent_const_vcas`, `descent_vs_*`, `finalapp_vcas`, `finalapp_vs`, `landing_speed`, and `landing_acceleration`; and the drag/thrust pages document a point-mass drag polar plus maximum/idle thrust functions, with idle thrust modeled at about 7% of max thrust at the same speed and altitude. ([OpenAP][1])

## 1) Pick the model scope

For final approach / approach management, I would use the continuous state

[
x = [,s,\ h,\ V,]
]

where:

* (s): along-track distance along the procedure centerline,
* (h): altitude,
* (V): **true airspeed**.

I recommend **TAS as the dynamic speed state**, even if your operational targets are in CAS. The reason is simple: drag and thrust are naturally written in TAS, while controllers and procedures are naturally written in CAS. So you simulate in TAS and expose CAS as an output:
[
V_{\text{cas}} = \text{tas2cas}(V,h,dT).
]

Latitude and longitude should be **derived** from a stored centerline geometry (r(s)), not integrated as primary states. That keeps the model compact and avoids lateral dynamics you do not care about.

For most approach segments, I would keep mass fixed:
[
m(t) \approx m_{\text{app}}
]
because over a short approach the fuel-burn effect on speed/drag is small compared with wind and procedure effects. If later needed, mass becomes a fourth state.

## 2) Inputs you need

You need three classes of inputs.

### A. Scenario inputs

These come from the procedure and weather:

* runway threshold and centerline geometry (r(s)),
* altitude profile (h_{\text{ref}}(s)),
* speed restrictions or target schedule nodes,
* wind field (w_\parallel(s,h,t)) along track,
* ISA temperature offset (dT) if you want non-standard atmosphere.

### B. Aircraft-specific parameters from OpenAP

From `prop.aircraft(typecode)`:

* wing area (S),
* clean drag coefficients (C_{D0,\text{clean}}) and (k_{\text{clean}}),
* limits like OEW, MLW, MTOW, VMO, MMO,
* default engine choice. ([OpenAP][1])

From `prop.engine(engine)` if you want thrust bounding:

* engine max thrust metadata and fuel-related coefficients. ([OpenAP][1])

From WRAP:

* `descent_const_vcas`
* `descent_cross_alt_concas`
* `descent_vs_concas`
* `descent_vs_post_concas`
* `finalapp_vcas`
* `finalapp_vs`
* `landing_speed`
* `landing_acceleration` ([OpenAP][2])

From `aero`:

* (\rho(h,dT)), (p(h,dT)), (T(h,dT)),
* `cas2tas`, `tas2cas`, `mach2tas`, `cas2mach`,
* constants such as (g_0), (\rho_0), (R), (a_0). ([OpenAP][3])

### C. Parameters you must define or calibrate yourself

OpenAP does **not** directly give these in the form you want:

* hidden drag “mode” parameters for clean / approach / final if you do not want explicit flap settings,
* speed-response lag (\tau_v),
* altitude-path capture gain (k_h),
* conservative tailwind used for backward speed-feasibility planning,
* optional extra deceleration from speedbrakes or operator technique.

That last category is where your simulator identity lives.

## 3) Use hidden modes, not explicit flap settings

You said you do not want to simulate flap handle positions or thrust lever settings. I agree.

Define a hidden discrete mode
[
q \in {\text{clean},\ \text{approach},\ \text{final}}.
]

Each mode has only these effects:

* a speed envelope,
* a vertical-rate envelope,
* an effective drag model,
* a speed-response lag.

OpenAP’s drag API has a `drag.clean()` case and also a `drag.nonclean()` path that explicitly takes flap angle and landing-gear state. Since you do not want those explicit cockpit variables in your simulator, the cleanest move is to absorb them into **effective mode-dependent drag coefficients** instead. ([OpenAP][4])

So instead of modeling flaps, use:
[
C_{D0}(q),\quad k(q),\quad \tau_v(q),\quad VS_{\min}(q),\quad VS_{\max}(q).
]

A practical initialization is:

* clean mode: use the clean drag coefficients from `prop.aircraft()`,
* approach/final modes: fit effective (C_{D0}) and (k) to observed deceleration behavior, or fit them to representative OpenAP nonclean drag outputs offline and then forget the flap angle variable entirely.

## 4) Atmosphere and speed conversions

At each time step:
[
(p,\rho,T_a) = \text{atmos}(h,dT),
]
[
V_{\text{cas}} = \text{tas2cas}(V,h,dT),
\qquad
V_{\text{ref,tas}} = \text{cas2tas}(V_{\text{ref,cas}},h,dT).
]

Using TAS as the state and CAS as the guidance variable is the least awkward architecture because:

* drag depends on air-relative speed,
* ATM procedures are usually stated in CAS,
* wind affects groundspeed, not aerodynamic drag directly. ([OpenAP][3])

## 5) Build the reference path first

### Vertical path

Represent the approach profile as either:
[
h_{\text{ref}}(s)
]
or equivalently a path angle
[
\gamma_{\text{ref}}(s) = \arctan!\left(-\frac{dh_{\text{ref}}}{ds}\right).
]

For an ILS-like glidepath this can be piecewise:

* level segments for altitude constraints,
* constant-angle descent on final.

### Speed schedule

Represent the speed schedule in **CAS**:
[
V_{\text{ref,cas}}(s).
]

Construct it from nodes such as:

* a descent CAS anchor from `descent_const_vcas`,
* a final-approach anchor from `finalapp_vcas`,
* a landing / threshold anchor from `landing_speed`,
* any procedure constraints you add. ([OpenAP][2])

If you want variability, WRAP gives each parameter as a distribution with default/min/max/statistical model, and the trajectory generator samples from those WRAP distributions for randomized trajectories. That makes WRAP a good source of Monte Carlo priors even if you do not use its simulator directly. ([OpenAP][2])

## 6) Make the speed profile feasible backward in distance

This is the step that stops the simulator from “teleporting” speed changes.

Take your target speed nodes in CAS, convert them to TAS on a planning altitude grid, and run a **backward feasibility pass** with a conservative along-track groundspeed (V_{gs,\text{plan}}), for example using forecast tailwind or a high-percentile tailwind.

A simple continuous form is:
[
\frac{dV_{\text{feas}}}{ds}
===========================

-\frac{a_{\text{dec,plan}}(V,h,q)}{V_{gs,\text{plan}}(s,h)},
]
integrated backward from runway to outer approach.

A simple discrete form is:
[
V_i^2 \le V_{i+1}^2 + 2,a_{\text{dec,plan},i},\Delta s.
]

Then set:
[
V_{\text{ref,cas}}(s) = \min\big(V_{\text{schedule,cas}}(s),,V_{\text{feas,cas}}(s)\big).
]

This single step is what creates realistic “slow to slow down in tailwind” behavior.

## 7) Derive drag from weight

Now the core aerodynamic piece.

Define weight
[
W = mg.
]

Assume no bank and small path-tracking error, so required lift is approximately
[
L \approx W\cos\gamma_{\text{ref}}.
]

Then
[
C_L = \frac{L}{\tfrac12 \rho V^2 S}.
]

Use an effective drag polar
[
C_D = C_{D0}(q) + k(q) C_L^2.
]

Then drag is
[
D = \tfrac12 \rho V^2 S C_D.
]

This is the same point-mass drag structure documented by OpenAP’s drag model. OpenAP states the drag module uses the drag polar coefficients (C_{d0}) and (k), computes (C_l) from lift, then computes (D = \tfrac12 \rho v^2 S C_d); it also notes that climb/descent geometry can change the lift estimate through flight-path angle / vertical speed. ([OpenAP][4])

If you expand it, you get the usual
[
D(V,h,m,q)
==========

\underbrace{\tfrac12 \rho S C_{D0}(q)}*{A(h,q)}V^2
+
\underbrace{\frac{2k(q),[mg\cos\gamma*{\text{ref}}]^2}{\rho S}}_{B(h,m,q)}
\frac{1}{V^2}.
]

So yes, the “parasite-like” term grows with (V^2), and the induced-like term shrinks with (1/V^2).

## 8) Decide how much thrust physics you want

You have two valid options.

### Option A: no explicit thrust variable

Use a direct limited-lag speed equation:
[
\dot V_{\text{cmd}} = \frac{V_{\text{ref,tas}} - V}{\tau_v(q)},
]
[
\dot V = \operatorname{sat}!\left(\dot V_{\text{cmd}},,-a_{\text{dec,max}}(V,h,m,q,\gamma_{\text{ref}}),,a_{\text{acc,max}}(V,h,m,q)\right).
]

Then define
[
a_{\text{dec,max}} \approx \frac{D - T_{\text{idle}}}{m} - g|\sin\gamma_{\text{ref}}|.
]

This is the simplest implementation.

### Option B: algebraic thrust, still no thrust “setting”

This is the more physical version, and it is still reduced-order.

Use the along-path force balance:
[
m\dot V = T - D - mg\sin\gamma_{\text{ref}}.
]

Define the thrust needed to achieve the lagged response:
[
T_{\text{cmd}}
==============

D + mg\sin\gamma_{\text{ref}}
+
m\frac{V_{\text{ref,tas}} - V}{\tau_v(q)}.
]

Then clip it:
[
T = \operatorname{clip}!\left(T_{\text{cmd}},\ T_{\text{idle}}(V,h),\ T_{\text{avail}}(V,h)\right).
]

Finally:
[
\dot V = \frac{T - D - mg\sin\gamma_{\text{ref}}}{m}.
]

This version never asks you to set thrust manually. Thrust is just an **internal algebraic variable** that enforces physical feasibility.

OpenAP’s thrust module provides maximum net thrust functions for takeoff/climb/cruise conditions and an idle-thrust estimate for descent, with idle thrust modeled as roughly 7% of max thrust at the same speed and altitude. ([OpenAP][4])

## 9) Vertical-motion model

For ATM realism, vertical motion should come from the path and groundspeed, not from a fixed vertical-rate target alone.

Let along-track groundspeed be
[
V_{gs} = V + w_\parallel(s,h,t)
]
if you ignore crosswind drift in the 1D model.

Then define feedforward vertical rate from the geometric path:
[
\dot h_{\text{ff}} = -V_{gs}\tan\gamma_{\text{ref}}(s).
]

Add a small path-capture correction:
[
\dot h_{\text{cmd}} = \dot h_{\text{ff}} + k_h,[h_{\text{ref}}(s)-h].
]

Then clip with plausible mode-dependent vertical-rate bounds:
[
\dot h = \operatorname{clip}!\left(\dot h_{\text{cmd}},,VS_{\min}(q),,VS_{\max}(q)\right).
]

WRAP gives you plausible descent and final-approach vertical-rate parameters such as `descent_vs_concas`, `descent_vs_post_concas`, and `finalapp_vs`, which are exactly the right source for those bounds. ([OpenAP][2])

This is where wind matters correctly:

* headwind lowers (V_{gs}), so the same path angle needs less vertical rate,
* tailwind raises (V_{gs}), so the same path angle needs more vertical rate.

## 10) Final system of differential equations

With fixed mass, the model I would recommend is:

[
\boxed{
\begin{aligned}
\dot s &= V + w_\parallel(s,h,t) [4pt]
\dot h &= \operatorname{clip}!\left(
-\big(V+w_\parallel\big)\tan\gamma_{\text{ref}}(s)
+
k_h,[h_{\text{ref}}(s)-h],,
VS_{\min}(q),,VS_{\max}(q)
\right) [6pt]
\dot V &= \frac{T - D - mg\sin\gamma_{\text{ref}}(s)}{m}
\end{aligned}}
]

with algebraic definitions

[
V_{\text{cas}} = \text{tas2cas}(V,h,dT),
\qquad
V_{\text{ref,tas}} = \text{cas2tas}(V_{\text{ref,cas}}(s),h,dT),
]

[
L \approx mg\cos\gamma_{\text{ref}}(s),
\qquad
C_L = \frac{L}{\tfrac12 \rho(h,dT) V^2 S},
]

[
C_D = C_{D0}(q) + k(q) C_L^2,
\qquad
D = \tfrac12 \rho(h,dT) V^2 S C_D,
]

[
T_{\text{cmd}} = D + mg\sin\gamma_{\text{ref}}(s) + m\frac{V_{\text{ref,tas}}-V}{\tau_v(q)},
]

[
T = \operatorname{clip}!\left(T_{\text{cmd}},\ T_{\text{idle}}(V,h),\ T_{\text{avail}}(V,h)\right).
]

That is the full reduced-order system.

If you want the simpler non-thrust form, eliminate (T) and use:
[
\boxed{
\dot V =
\operatorname{sat}!\left(
\frac{V_{\text{ref,tas}}-V}{\tau_v(q)},
,-a_{\text{dec,max}}(V,h,m,q),,a_{\text{acc,max}}(V,h,m,q)
\right)
}
]
with
[
a_{\text{dec,max}} \approx \frac{D-T_{\text{idle}}}{m}-g|\sin\gamma_{\text{ref}}|.
]

## 11) How to choose every parameter

Here is the practical assignment.

### Taken directly from OpenAP

Use:

* (S), (C_{D0,\text{clean}}), (k_{\text{clean}}), OEW/MLW/MTOW, VMO/MMO from `prop.aircraft()`,
* engine default / max thrust metadata from `prop.engine()`,
* atmosphere and CAS/TAS conversion from `aero`,
* descent/final/landing WRAP anchors from `WRAP`. ([OpenAP][1])

### Chosen by you once

Use:

* (m_{\text{app}}): a representative approach mass, often between OEW+payload and MLW,
* (k_h): vertical path capture gain,
* (q)-transition logic: usually based on distance-to-runway and speed threshold,
* conservative planning tailwind.

### Calibrated from data

Fit:

* (C_{D0}(\text{approach})), (k(\text{approach})),
* (C_{D0}(\text{final})), (k(\text{final})),
* (\tau_v(\text{clean})), (\tau_v(\text{approach})), (\tau_v(\text{final})).

That is the minimal set that controls realistic slowing-down.

## 12) Mode-transition logic

Keep it simple. For example:

* `clean` outside the terminal slowdown segment,
* `approach` when (V_{\text{cas}}) gets within some margin of the approach anchor or when distance to threshold drops below a chosen gate,
* `final` when intercepting the final glide segment or when near stabilized approach conditions.

The trigger itself is not sacred. The key is that each mode changes only:

* the speed envelope,
* the drag model,
* the lag (\tau_v),
* the vertical-rate bounds.

## 13) What WRAP does for you, and what it still does not

WRAP is ideal for:

* nominal speed anchors,
* nominal vertical-rate envelopes,
* randomization priors for Monte Carlo. ([OpenAP][2])

WRAP still does **not** by itself give you:

* a wind-coupled deceleration feasibility model,
* a hidden-mode drag law,
* a response lag (\tau_v),
* your route-relative ODE integrator.

So the best architecture is still:

[
\text{WRAP envelopes} + \text{your path model} + \text{your drag/lag feasibility layer}.
]

## 14) Numerical integration loop

At each step (t_k \to t_{k+1}):

1. Read (s_k,h_k,V_k).
2. Compute (\rho,p,T_a) and (V_{\text{cas},k}).
3. Evaluate mode (q_k).
4. Compute (V_{\text{ref,cas}}(s_k)), then (V_{\text{ref,tas}}).
5. Compute (\gamma_{\text{ref}}(s_k)), (h_{\text{ref}}(s_k)), and (w_\parallel(s_k,h_k,t_k)).
6. Compute (D_k).
7. Compute (T_k) or directly compute limited (\dot V_k).
8. Compute (\dot s_k,\dot h_k,\dot V_k).
9. Integrate with RK4 or even Euler if your step is small enough.
10. Map (s_{k+1}) to lat/lon via the centerline geometry.

## 15) Minimal calibration strategy

If I had to get this working fast, I would calibrate in this order:

1. Match threshold/final CAS and GS distributions.
2. Match altitude-vs-distance on final.
3. Match where deceleration starts in headwind/tailwind cases.
4. Match time-to-lose 20 kt and 40 kt in clean-to-approach slowdown.
5. Only then refine mode-dependent drag.

That ordering gets you ATM realism sooner than trying to identify exact aerodynamic truth first.

## 16) The shortest usable version

If you want the most compact model that is still defensible, use:

[
\dot s = V + w_\parallel,
]

[
\dot h =
\operatorname{clip}!\left(
-(V+w_\parallel)\tan\gamma_{\text{ref}} + k_h(h_{\text{ref}}-h),
VS_{\min},VS_{\max}
\right),
]

[
\dot V =
\operatorname{sat}!\left(
\frac{\text{cas2tas}(V_{\text{ref,cas}},h,dT)-V}{\tau_v(q)},
-a_{\text{dec,max}}(V,h,q),
a_{\text{acc,max}}(V,h,q)
\right),
]

with
[
a_{\text{dec,max}}(V,h,q)
\approx
\frac{
\tfrac12\rho V^2 S!\left(C_{D0}(q)+k(q)C_L^2\right)
---------------------------------------------------

T_{\text{idle}}(V,h)
}{m}
-g|\sin\gamma_{\text{ref}}|.
]

That is probably the best balance of realism, simplicity, and computational cost for your problem.

The most useful next artifact would be a parameter sheet with exact symbols, units, OpenAP source field names, and default values for one aircraft such as A320 or B738.

[1]: https://openap.dev/aircraft_engine.html "1   Aircraft and engines – The OpenAP Handbook"
[2]: https://openap.dev/kinematic.html "3   Kinematic models – The OpenAP Handbook"
[3]: https://openap.dev/api/aero.html "21  ️ aero – The OpenAP Handbook"
[4]: https://openap.dev/drag_thrust.html "2  ☯️ Drag and thrust – The OpenAP Handbook"
