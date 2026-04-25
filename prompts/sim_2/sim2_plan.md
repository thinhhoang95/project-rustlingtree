# SIMAP VNAV - Authoritative Design

One important caveat first: with only a touchdown state and path constraints, **TOD is not unique**. To get a unique TOD, you need either an **upstream terminal condition** such as cruise altitude/speed, or an **objective** such as “stay as close to idle as possible while maximizing upstream distance.” Also, for descent planning, it is usually better to anchor the backward solve at **threshold crossing / stabilized approach** rather than wheel touchdown, because flare is a separate phase.

You also need a few aircraft parameters beyond your current list: at minimum **mass**, **wing reference area**, an atmosphere model ( \rho(h) ), configuration-dependent drag-polar coefficients, and a thrust model or thrust bounds. Public aircraft-performance models such as BADA parameterize exactly those items: mass, flight envelope, aerodynamics, engine thrust, and fuel consumption. ([EUROCONTROL][1])

## 1) Longitudinal aircraft dynamics

Use a 2D point-mass longitudinal model with states
[
s(t),\ h(t),\ V(t),\ \gamma(t),\ m(t),
]
where (s) is **along-track distance to threshold**, positive upstream, so the aircraft moves with ( \dot s < 0 ).

The time-domain equations are
[
\dot s = -V\cos\gamma,
]
[
\dot h = V\sin\gamma,
]
[
m\dot V = T - D - mg\sin\gamma,
]
[
mV\dot\gamma = L - mg\cos\gamma,
]
[
\dot m = -\dot m_f(T,h,V,c).
]

Aerodynamic lift and drag are
[
q = \tfrac12 \rho(h)V^2,\qquad L = qSC_L,\qquad D = qSC_D.
]

For the drag model, use a **polar drag law** with configuration-dependent coefficients:
[
C_D = C_{D0}(c) + C_{D2}(c),C_L^2 + \Delta C_{D,\text{gear}}(c).
]
That is the standard form used in performance models such as BADA, which uses separate clean / approach / landing coefficients and an added landing-gear drag term. NASA’s lift and drag equations use the same (qSC) structure. ([NASA][2])

Thrust must be bounded:
[
T_{\min}(h,V,c)\ \le\ T\ \le\ T_{\max}(h,V,c).
]
A BADA-style implementation gives you maximum climb/takeoff thrust, maximum cruise thrust, and descent thrust as functions of engine type, altitude, speed, and temperature deviation, which is a good template for simulator bounds. ([EUROCONTROL][1])

## 2) Convert it into a backward constrained ODE problem

Since your constraints are indexed by **distance from threshold**, the natural independent variable is (s), not time.

Using
[
\dot s = -V\cos\gamma,
]
you can rewrite the system as derivatives with respect to (s):

[
\frac{dh}{ds} = -\tan\gamma,
]

[
\frac{dV}{ds}
=============

-\frac{\frac{T-D}{m} - g\sin\gamma}{V\cos\gamma},
]

[
\frac{d\gamma}{ds}
==================

-\frac{L-mg\cos\gamma}{mV^2\cos\gamma},
]

[
\frac{dt}{ds} = -\frac{1}{V\cos\gamma},
]

[
\frac{dm}{ds} = \frac{\dot m_f}{V\cos\gamma}.
]

That is the exact backward-in-distance form.

For a **descent planner**, I would usually reduce this to a quasi-steady model by assuming
[
mV\dot\gamma \approx 0,
\qquad\Rightarrow\qquad
L \approx mg\cos\gamma.
]
Then
[
C_L = \frac{2mg\cos\gamma}{\rho V^2 S},
]
and the drag polar closes algebraically. BADA itself uses a slightly simpler assumption for (C_L), effectively taking flight-path angle as zero in that step; keeping the (\cos\gamma) term is a small refinement that is more self-consistent for your simulator. ([EUROCONTROL][1])

So the **recommended planning model** is:

State:
[
y(s) = [h(s),,V(s),,m(s),,t(s)]
]

Control:
[
u(s) = [\gamma(s),,T(s)]
]

Dynamics:
[
h' = -\tan\gamma,
]
[
V' = -\frac{\frac{T-D(h,V,\gamma,c(s))}{m} - g\sin\gamma}{V\cos\gamma},
]
[
m' = \frac{\dot m_f(T,h,V,c(s))}{V\cos\gamma},
]
[
t' = -\frac{1}{V\cos\gamma}.
]

## 3) Path constraints

Let (c(s)\in{\text{clean},\text{approach},\text{final}}) be your configuration schedule.

At each (s), define the feasible speed interval as the intersection of everything that applies (given speed bounds and configuration stall bounds):
[
V_L(s)=
\max!\Big(
V_{\text{cfg,min}}(c(s)),
V_{\text{spd,min}}(s),
V_{\text{stall-margin}}(h,m,c(s))
\Big),
]
[
V_U(s)=
\min!\Big(
V_{\text{cfg,max}}(c(s)),
V_{\text{spd,max}}(s)
\Big).
]

Altitude constraints are
[
h_L(s)\le h(s)\le h_U(s).
]

Thrust constraints are
[
T_{\min}(h,V,c(s))\le T(s)\le T_{\max}(h,V,c(s)).
]

You can also add
[
|\gamma(s)| \le \gamma_{\max},
\qquad
0 \le C_L(h,V,\gamma,c(s)) \le C_{L,\max}(c(s)).
]

## 4) Boundary conditions and what “TOD” means mathematically

Use (s=0) at the threshold-side anchor point.

A practical boundary is
[
h(0)=h_{\text{thr}},
\qquad
V(0)=V_{\text{app}},
]
and optionally
[
\gamma(0)=\gamma_{\text{app}}
]
if you want to pin the final approach angle.

Then define TOD as a **free upstream boundary** (s_{\text{TOD}}) with terminal conditions such as
[
h(s_{\text{TOD}})=h_{\text{crz}},
]
[
V(s_{\text{TOD}})\in [V_{\text{crz,min}},V_{\text{crz,max}}],
]
and optionally
[
\gamma(s_{\text{TOD}})=0
]
if you want level flight at the descent start.

So the full problem is a **free-boundary constrained ODE / optimal-control problem**:
[
y'(s)=f(s,y,u),
\quad s\in[0,s_{\text{TOD}}],
]
subject to path constraints and endpoint constraints.

## 5) Objective that unifies the “idle” and “geometric” segments

This is the key unification.

Instead of hard-coding “idle segment” and “geometric segment,” solve one problem with an objective like
[
J
=

-w_s,s_{\text{TOD}}
+
w_T\int_0^{s_{\text{TOD}}}!!\big(T(s)-T_{\text{idle}}(h,V,c)\big)^2,ds
+
w_\gamma\int_0^{s_{\text{TOD}}}!!(\gamma'(s))^2,ds
+
w_\epsilon\int_0^{s_{\text{TOD}}}!!|\epsilon(s)|^2,ds,
]
where ( \epsilon ) are optional slack variables for soft constraints.

Interpretation:

* the first term pushes TOD as far upstream as possible,
* the second makes the descent as close to idle as possible,
* the third smooths the path,
* the slack term prevents solver failure when the constraint set is slightly inconsistent.

With this setup, the solver naturally produces:

* **idle-like descent** where feasible,
* **path adjustments** where constraints force them,
* **more thrust** only when physically required.

So the two classical “segments” fall out of one optimization, instead of being coded separately.

## 6) Best Python method

For the **full constrained problem**, I would **not** start with `solve_bvp`.

`solve_bvp` is designed for first-order two-point boundary value problems, and it can handle free parameters, but once you add lots of inequality/path constraints and bounded controls, it gets awkward. SciPy’s constrained optimizers are a better fit: `scipy.optimize.minimize(method="trust-constr")` supports variable bounds plus nonlinear constraints via `Bounds`, `LinearConstraint`, and `NonlinearConstraint`. `solve_ivp` is still very useful for propagation and validation, and it supports event handling plus stiff solvers such as `Radau` and `BDF`. ([SciPy Documentation][3])

So the recommended recipe is:

1. Discretize (s\in[0,s_{\text{TOD}}]) on (N) nodes.
2. Make the node values of (h_i,V_i,\gamma_i,T_i) decision variables, plus the free scalar (s_{\text{TOD}}).
3. Enforce the dynamics with collocation defects.
4. Enforce speed, altitude, thrust, and configuration constraints at each node.
5. Solve with `trust-constr`.
6. Re-propagate the resulting profile with `solve_ivp` to verify it.

## 7) Concrete transcription

Let ( \sigma\in[0,1] ) be a normalized mesh and let
[
s = \sigma,s_{\text{TOD}}.
]
That makes the free endpoint (s_{\text{TOD}}) easy to optimize.

At node (i), define
[
x_i = [h_i,\ V_i,\ m_i,\ t_i],\qquad u_i=[\gamma_i,\ T_i].
]

Use trapezoidal defects:
[
x_{i+1} - x_i - \frac{\Delta s}{2}
\Big(f(s_i,x_i,u_i)+f(s_{i+1},x_{i+1},u_{i+1})\Big)=0.
]

That gives you equality constraints. Then add:

* endpoint constraints at (i=0) and (i=N),
* box constraints on (V_i), (h_i), (T_i),
* nonlinear constraints for (C_L), stall margin, etc.

## 8) Minimal implementation skeleton

```python
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint

g = 9.80665

def rho_isa(h):
    # user atmosphere model
    ...

def cfg_of_s(s):
    # clean / approach / final
    ...

def drag_force(h, V, gamma, m, cfg, S, cd0, cd2, dcd_gear):
    rho = rho_isa(h)
    q = 0.5 * rho * V**2
    CL = 2.0 * m * g * np.cos(gamma) / (rho * V**2 * S)
    CD = cd0[cfg] + cd2[cfg] * CL**2 + dcd_gear[cfg]
    D = q * S * CD
    return D, CL, CD

def f_state(s, x, u, params):
    h, V, m, t = x
    gamma, T = u
    cfg = cfg_of_s(s)
    D, CL, CD = drag_force(h, V, gamma, m, cfg,
                           params["S"], params["cd0"], params["cd2"], params["dcd_gear"])
    dhds = -np.tan(gamma)
    dVds = -(((T - D) / m) - g * np.sin(gamma)) / (V * np.cos(gamma))
    dmds = params["fuel_flow"](T, h, V, cfg) / (V * np.cos(gamma))
    dtds = -1.0 / (V * np.cos(gamma))
    return np.array([dhds, dVds, dmds, dtds]), CL

def objective(z, data):
    # unpack z -> s_TOD, node states, node controls
    # objective = -w_s*s_TOD + w_T*sum((T-T_idle)^2)*ds + smoothness penalties
    ...

def defect_constraints(z, data):
    # trapezoidal collocation defects
    ...

def path_constraints(z, data):
    # stack V-Vmin, Vmax-V, h-hmin, hmax-h, T-Tmin, Tmax-T, CLmax-CL
    ...

# bounds: h, V, gamma, T, s_TOD
bounds = Bounds(lb, ub)

eq_cons = NonlinearConstraint(defect_constraints, 0.0, 0.0)
ineq_cons = NonlinearConstraint(path_constraints, 0.0, np.inf)

res = minimize(objective, z0, args=(data,),
               method="trust-constr",
               bounds=bounds,
               constraints=[eq_cons, ineq_cons])
```

## 9) Practical initialization

A good initial guess is:

* (h(s)): straight-line descent from threshold to cruise altitude,
* (V(s)): piecewise-linear schedule through your speed constraints,
* (\gamma(s)): constant (-3^\circ) guess,
* (T(s)): idle or near-idle everywhere,
* (s_{\text{TOD}}): from a crude geometric estimate,
  [
  s_{\text{TOD},0} \approx \frac{h_{\text{crz}}-h_{\text{thr}}}{\tan 3^\circ}.
  ]

Then let the optimizer bend that initial path to satisfy the full drag/thrust/speed envelope.

## 10) Bottom line

The clean recipe is:

* use the **full point-mass longitudinal model** for the aircraft,
* use a **distance-based quasi-steady reduced model** for backward descent planning,
* model drag with a **configuration-dependent quadratic polar**,
* impose **bounded thrust** explicitly,
* formulate TOD as a **free-boundary constrained optimal-control problem**,
* solve it in Python with **direct collocation + `trust-constr`**,
* and validate the result by replaying it with `solve_ivp`. ([SciPy Documentation][3])

[1]: https://www.eurocontrol.int/archive_download/all/node/9690 "SATT-EEC-Technical-Report-2010-003"
[2]: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation/ "Drag Equation | Glenn Research Center | NASA"
[3]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html "solve_bvp — SciPy v1.17.0 Manual"
