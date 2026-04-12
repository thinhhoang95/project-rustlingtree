The **minimal lateral model** is:

* **state:** heading ( \psi )
* **optional hidden fast state:** bank angle ( \phi )

I would strongly recommend keeping ( \phi ) as a **hidden internal state**, even if you do not expose it as part of the research state. Without it, turns start and stop with an unrealistic instantaneous curvature jump.

## 1) The core coordinated-turn equations

Assume coordinated flight, so sideslip is negligible. The FAA defines coordinated flight as having the nose aligned with the relative wind and the slip/skid ball centered. ([faa.gov][1])

Then the standard coordinated-turn kinematics are:

[
\dot{\psi} = \frac{g \tan\phi}{V}
]

and equivalently

[
R_{\text{air}} = \frac{V^2}{g \tan\phi},
\qquad
\phi = \arctan!\left(\frac{V^2}{gR_{\text{air}}}\right).
]

A NASA trajectory-generation paper states the same bank-angle/radius relation for a coordinated turn. 

Here:

* (V) = **TAS**
* (R_{\text{air}}) = turn radius in the airmass
* (\psi) = heading

So if you only care about heading, this is the minimal dynamic law.

## 2) Where weight enters

This is the important part: **weight does not directly appear in**
[
R=\frac{V^2}{g\tan\phi}.
]

For a given airspeed and bank angle, a heavier and lighter aircraft have the **same kinematic turn radius** under the coordinated-turn assumption.

Weight enters **indirectly** by limiting the bank angle you can safely or operationally use, because in a coordinated level turn the load factor increases with bank, and stall speed rises with load factor. The FAA notes that in a coordinated level 60° bank turn the load factor is 2G, and the stall speed is about 41% higher than the 1G stall speed. ([faa.gov][1])

So for your simulator, the clean way to model “turn radius as a function of weight and configuration mode” is:

* keep the turn law itself kinematic,
* make **maximum usable bank angle** depend on weight, speed, and mode.

That is the right abstraction.

## 3) Configuration mode enters through stall speed and bank limits

Let your hidden mode be:

[
q \in {\text{clean},\ \text{approach},\ \text{final}}.
]

For each mode, define a 1G reference stall speed in CAS:
[
V_{s,1g}^{\text{ref}}(q)
]

at some reference weight (W_{\text{ref}}).

Then scale it with weight using the standard square-root law:

[
V_{s,1g}(q,W)
=============

V_{s,1g}^{\text{ref}}(q)\sqrt{\frac{W}{W_{\text{ref}}}}.
]

In a coordinated turn, load factor is

[
n = \frac{1}{\cos\phi},
]

so the turning stall speed becomes

[
V_{s,\text{turn}}(q,W,\phi)
===========================

# V_{s,1g}(q,W)\sqrt{n}

\frac{V_{s,1g}(q,W)}{\sqrt{\cos\phi}}.
]

This is exactly the mechanism behind the FAA’s “stall speed rises in bank” explanation. ([faa.gov][1])

Now impose a safety margin (\lambda > 1), for example a maneuvering margin above stall:

[
V_{\text{cas}} \ge \lambda , V_{s,\text{turn}}.
]

Rearranging gives a **stall-limited bank angle**:

[
\phi_{\max,\text{stall}}
========================

\arccos!\left(\left(\frac{\lambda V_{s,1g}(q,W)}{V_{\text{cas}}}\right)^2\right).
]

So:

* **heavier aircraft** (\Rightarrow) larger (V_s) (\Rightarrow) smaller allowable bank at the same speed,
* **clean mode** usually has higher stall speed than landing configuration (\Rightarrow) smaller bank at the same speed,
* **final mode** may also have a stricter operational bank cap even if lift is available.

That is how weight and configuration should influence turn radius in your model.

## 4) Final bank-angle limit

Use the most restrictive of three limits:

[
\phi_{\max}
===========

\min!\Big(
\phi_{\max,\text{comfort}}(q),,
\phi_{\max,\text{procedure}}(q),,
\phi_{\max,\text{stall}}(q,W,V_{\text{cas}})
\Big).
]

Where:

* ( \phi_{\max,\text{comfort}} ): passenger-comfort or ATM realism cap
* ( \phi_{\max,\text{procedure}} ): your own mode-dependent cap
* ( \phi_{\max,\text{stall}} ): the physics-based cap above

Then command

[
\phi = \operatorname{clip}(\phi_{\text{req}}, -\phi_{\max}, \phi_{\max}).
]

## 5) How to compute the required bank

There are two equally good ways.

### If your lateral guidance gives desired curvature

If your route centerline gives desired curvature ( \kappa_{\text{cmd}} = 1/R_{\text{cmd}} ), then

[
\phi_{\text{req}} = \arctan!\left(\frac{V^2 \kappa_{\text{cmd}}}{g}\right).
]

### If your lateral guidance gives desired heading rate

If you command heading directly,

[
\phi_{\text{req}} = \arctan!\left(\frac{V \dot{\psi}_{\text{cmd}}}{g}\right).
]

Then propagate heading using

[
\dot{\psi} = \frac{g\tan\phi}{V}.
]

That is the simplest closed loop.

## 6) Should you add bank as a hidden state?

I would say yes.

If you set ( \phi = \phi_{\text{req}} ) algebraically, the aircraft begins turning instantly. That is usually too sharp.

A very cheap fix is:

[
\dot{\phi} = \operatorname{sat}!\left(\frac{\phi_{\text{req}}-\phi}{\tau_\phi(q)},,-p_{\max}(q),,p_{\max}(q)\right)
]

then

[
\dot{\psi} = \frac{g\tan\phi}{V}.
]

This adds almost no computational cost, but it gives:

* finite roll-in,
* finite roll-out,
* realistic turn capture,
* more realistic overshoot behavior.

So the practical lateral state set becomes:

[
x_{\text{lat}} = [\psi,\phi]
]

with only ( \psi ) exposed if you want.

## 7) What about wind?

This depends on whether you care about **heading** or **ground track**.

### If you only care about heading

Then use the airmass relation:

[
\dot{\psi} = \frac{g\tan\phi}{V}.
]

Wind does not directly enter the heading-turn law.

### If you care about the actual path over ground

Then you need position too:

[
\dot{x} = V\cos\psi + w_x,
\qquad
\dot{y} = V\sin\psi + w_y.
]

Then the **ground-track** radius is not exactly the same as the air-relative turn radius when wind is present. The NASA route-mapping paper notes that along-track wind changes groundspeed, and crosswind effects are more subtle. 

For ATM path geometry, this matters if you want realistic fly-by turns in wind.

## 8) My recommended lateral model for you

I would use this:

[
\boxed{
\begin{aligned}
\dot{\phi} &=
\operatorname{sat}!\left(
\frac{\phi_{\text{req}}-\phi}{\tau_\phi(q)},
-p_{\max}(q),,p_{\max}(q)
\right) [4pt]
\dot{\psi} &= \frac{g\tan\phi}{V}
\end{aligned}}
]

with

[
\phi_{\text{req}} = \arctan!\left(\frac{V^2\kappa_{\text{cmd}}}{g}\right)
]

and

[
\phi_{\max}
===========

\min!\Big(
\phi_{\max,\text{comfort}}(q),,
\phi_{\max,\text{procedure}}(q),,
\arccos!\left(\left(\frac{\lambda V_{s,1g}^{\text{ref}}(q)\sqrt{W/W_{\text{ref}}}}{V_{\text{cas}}}\right)^2\right)
\Big).
]

Then clip:

[
\phi_{\text{req}} \leftarrow \operatorname{clip}(\phi_{\text{req}},-\phi_{\max},\phi_{\max}).
]

## 9) Interpretation

So the answer to your question is:

* **heading is enough as the exposed lateral state**
* **bank angle should be kept as a hidden internal state**
* **turn radius is not directly a function of weight**
* **weight and configuration affect turn radius by limiting bank angle through stall margin and operational caps**

That is the most defensible reduced-order lateral model for your simulator.

[1]: https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/airplane_handbook/06_afh_ch5.pdf "Airplane Flying Handbook (3C) Chapter 5"
