Yes: for the **trombone pattern**, the SVD does need to be rethought, but not thrown away. The right change is to move SVD **after lag registration**, so SVD explains the **shape of the U-turn**, while a separate scalar explains **where along the downwind leg that U-turn occurs**.

Your current HLLRD model is fixed-support and fixed-coordinate: each event has an interval (I_k), a local rank-2 basis (V_k) supported in that interval, and per-flight coefficients (Z_k), so the reconstruction is essentially (X_{\text{centered}} \approx \sum_k Z_k V_k^T). That is a good model for localized amplitude/shape variation, but a trombone is partly a **phase variation**: the same shape appears at different station locations.  

## The core problem

If the residual pattern for flight (i) is approximately

[
r_i(s) = g(s-\tau_i) + \epsilon_i(s),
]

then the important parameter is not a new basis coefficient. It is the lag

[
\tau_i,
]

which says how far downwind the aircraft continued before making essentially the same turn.

A fixed-window SVD sees all of these shifted copies in the same coordinate system. For small shifts, it starts to represent the family as

[
g(s-\tau_i)
\approx
g(s) - \tau_i g'(s) + \frac{\tau_i^2}{2}g''(s) - \cdots.
]

For larger/random shifts, the covariance becomes close to a translation/convolution operator, so the principal components become sinusoidal/cosine-like modes rather than “the U-turn shape.” That is exactly the collapse you described.

This is closely related to the amplitude/phase separation problem in functional data analysis: phase variation can be removed by alignment, but removing it entirely can also discard meaningful timing/location information; newer methods explicitly model both amplitude and phase rather than forcing ordinary PCA/SVD to explain both at once. ([arXiv][1]) Shift registration is the simplest version of this idea: align functions by a translation (\tilde{x}_i(t)=x_i(t+\delta_i)) before comparing their shapes. ([Journal of Statistical Software][2])

## What the trombone event should look like

For non-trombone events, keep your current localized low-rank model.

For trombone events, use a **shifted localized low-rank model**:

[
R_i(s)
\approx
C_{\text{in}}(s;\tau_i)
+
m(s;\tau_i)\sum_{\ell=1}^{r} z_{i\ell} v_\ell(s-\tau_i)
+
C_{\text{out}}(s;\tau_i)
+
\epsilon_i(s).
]

Equivalently:

[
R_i
\approx
C_{\text{in}}(\tau_i)
+
S_{\tau_i} B z_i
+
C_{\text{out}}(\tau_i)
+
\epsilon_i.
]

Where:

[
\tau_i
]

is the trombone lag, preferably in meters, nautical miles, seconds, or station units;

[
B = [v_1,\ldots,v_r]
]

is the canonical U-turn shape basis in a **turn-centered coordinate**;

[
z_i
]

contains residual turn-shape coefficients after the lag has been removed;

[
S_{\tau_i}
]

is a row-specific shift/interpolation operator; and

[
C_{\text{in}}, C_{\text{out}}
]

are stitching terms that keep the entry and exit geometry smooth.

So the compact optimizer vector should become something like:

[
y_i^{\text{trombone}}
=====================

(\text{active}*i,\ \Delta d_i,\ z*{i1},\ z_{i2}),
]

or, if the registered U-turn is nearly rank-1,

[
y_i^{\text{trombone}}
=====================

(\text{active}_i,\ \Delta d_i,\ q_i).
]

Here (\Delta d_i) is the downwind extension / delay, and (q_i) is the turn-shape parameter. That is much closer to what operators actually do: “extend downwind farther, then make the turn,” rather than “combine six cosine-like deformation modes.”

## How to reinterpret the event window

The trombone needs two windows, not one.

The first is the **envelope window**:

[
\Omega = [s_{\min}, s_{\max}],
]

large enough to contain the earliest and latest possible U-turn placements, plus entry/exit buffers.

The second is the **canonical turn window**:

[
J = [u_0,u_1],
]

centered on the U-turn itself, after shifting each flight by (\tau_i).

The mistake would be to run ordinary SVD on (R[:,\Omega]). That asks SVD to explain both lag and shape. Instead, use (\Omega) only to discover and bound the trombone event, then build a registered matrix

[
Y_i(u) = R_i(s_0 + \tau_i + u), \qquad u \in J,
]

and run SVD on (Y), not on the unregistered envelope.

So:

[
\text{wide envelope} \neq \text{wide fixed SVD basis}.
]

The envelope can be broad; the shape basis should remain compact.

## Relation to your proposed model

You suggested something like:

[
\frac{dx}{d\theta}\bigg|*{\text{in}}\theta*{\text{in}}
+
ZV^T
+
\frac{dx}{d\theta}\bigg|*{\text{out}}\theta*{\text{out}}.
]

That is directionally right if (\theta_{\text{in}}) and (\theta_{\text{out}}) are being used as **phase/anchor parameters**, but I would not let (ZV^T) live in the original fixed station coordinate for trombones.

A safer interpretation is:

[
C_{\text{in}}(\tau_i,\eta_i)
+
S_{\tau_i}(B z_i)
+
C_{\text{out}}(\tau_i,\eta_i).
]

The derivative terms are useful as a **local linearization** of moving the entry and exit boundaries. But if the lag range is operationally meaningful, meaning several stations or more, the exact shift operator (S_{\tau_i}) is better than a derivative expansion. Otherwise the model reintroduces the same “basis proliferation” problem.

## Practical fitting algorithm

A good V2 trombone fitter could be an alternate candidate type next to your current fixed-window candidates.

For each broad candidate envelope (\Omega):

1. **Initialize the lag.** Estimate a turn landmark per flight: turn-start, max-curvature point, base-turn apex, heading-crossing, or final-intercept landmark. Set

   [
   \tau_i = p_i - p_{\text{ref}}.
   ]

2. **Register the residuals.** Build

   [
   Y_i(u)=R_i(p_{\text{ref}}+\tau_i+u)
   ]

   using interpolation and masks for clipped edges.

3. **Run SVD in the registered coordinate.**

   [
   Y \approx Z B^T.
   ]

   This is where SVD still belongs.

4. **Refit each flight.** For each active flight, solve

   [
   \min_{\tau_i,z_i}
   \left|
   R_i - S_{\tau_i}Bz_i
   \right|^2
   +
   \lambda_\tau(\tau_i-\tau_{i,0})^2
   +
   \lambda_z|z_i|^2.
   ]

   In practice, (\tau_i) can be a one-dimensional grid search plus ridge least squares for (z_i).

5. **Update the registered SVD.** Repeat registration → SVD → lag refit a few times.

6. **Score the event by registered gain**, not ordinary fixed-window rank-2 gain:

   [
   \text{gain}
   ===========

   \sum_i
   \left(
   |R_i[\Omega]|^2
   ---------------

   |R_i[\Omega] - S_{\tau_i}Bz_i|^2
   \right).
   ]

7. **Activate flights** if either the lag is operationally meaningful, the registered shape amplitude is meaningful, or both:

   [
   |\tau_i| > \tau_{\min}
   \quad\text{or}\quad
   |z_i|_2 > \tau_z.
   ]

This keeps the current HLLRD spirit: localized, interpretable, compact, greedy-compatible. But it adds one nonlinear per-flight parameter for the special case where linear SVD is the wrong representation.

## Consequences for your artifact and transform design

This is the main architectural implication: a trombone event cannot be represented cleanly by one global dictionary matrix

[
D \in \mathbb{R}^{M \times 2K}
]

with a purely linear coefficient refit. Your current artifact and transform logic assume that the learned dictionary is fixed in station coordinates and that new data can be transformed by refitting coefficients against that dictionary. 

For trombones, the event should be stored as a different event type:

```text
event_type: shifted_low_rank
envelope_interval: [start_min, end_max)
canonical_interval: [u_start, u_end)
lag_reference_station: s_ref
basis_canonical: shape (L_canonical, r)
lag_by_flight: shape (n,)
coefficients: shape (n, r)
active_mask: shape (n,)
```

For transform on new flights, you would not only project onto a fixed dictionary. You would estimate

[
(\tau_i,z_i)
]

for each new flight using the saved canonical basis. Standard events remain linear; trombone events require a small one-dimensional nonlinear search.

## Important warning about normal residuals only

Your V1 matrix uses normal-direction residuals at reference stations and explicitly avoids confusing along-track mismatch with lateral deviation. That is good for lateral deformations, but a trombone is partly an along-track / phase / path-length phenomenon. 

So for trombone detection, I would not rely only on normal residuals. Estimate (\tau_i) from raw geometry or additional channels such as:

[
\text{heading}(s),\quad
\text{curvature}(s),\quad
\text{cumulative path length},\quad
\text{turn-start station},\quad
\text{final-intercept station}.
]

Then use normal residuals mainly to learn the registered lateral shape basis. In other words:

[
\text{raw XY / heading / curvature} \rightarrow \tau_i,
]

[
\text{registered normal residuals} \rightarrow z_i.
]

That separation is likely to be much more operator-conformant.

## Suggested decision rule

You can automatically decide whether an event is an ordinary local low-rank event or a trombone event by comparing rank spectra.

For a candidate envelope (\Omega):

1. Run ordinary fixed-window SVD on (R[:,\Omega]).
2. Estimate lags (\tau_i), register, and run SVD on (Y).
3. If registered rank-1 or rank-2 explains much more energy and (z_i) is no longer strongly correlated with (\tau_i), classify it as a trombone.

Operationally, a healthy trombone fit should have these properties:

[
\tau_i \text{ changes turn location},
]

[
z_i \text{ changes turn shape},
]

and varying (\tau_i) while holding (z_i) fixed should slide the U-turn downwind without changing its character.

So the short answer is: **yes, SVD should be rethought for trombones, but only by making it lag-aware.** Keep SVD as the shape learner, but do not ask fixed-window SVD to represent random lag. The trombone model should be “one phase parameter plus one or two registered shape parameters,” not a larger bank of stationary basis functions.

[1]: https://arxiv.org/abs/1603.01775 "[1603.01775] Combined Analysis of Amplitude and Phase Variations in Functional Data"
[2]: https://www.jstatsoft.org/article/download/v109i02/4562 "scikit-fda: A Python Package for Functional Data Analysis"
