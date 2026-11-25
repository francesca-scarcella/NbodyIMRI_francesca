# NbodyIMRI 0.2.0: force-split sub-steps (2026-11-25)

Starting point: `NbodyIMRI_francesca`, branch `inner-substeps`, commit d5394c8 (Plummer softening of M1).
The shadow black holes and the old sub-step scheme are removed; there is one scheme.

## The scheme (run_simulation with N_step_partition > 1)

    K_out(dt/2)  [ short step of the inner partition ]^N  D_out(dt)  K_out(dt/2)

* Inner partition: the two black holes and the particles within `r_partition` of M1 or `r_partition2` of M2.
* `K_out`: kick between the OUTER particles and the black holes, in both directions, at the same instant.
* Short step: 4th-order (PEFRL) step of length dt/N of the inner particles and the black holes
  (black holes on inner particles, inner particles on black holes, black holes on each other).
* `D_out`: free drift of the outer particles.

Code: `simulator.full_step_split`. Version marker: `simulator.SCHEME == "force_split"`.

## New arguments of `simulator.simulator`

* `r_partition2` : sub-step radius around M2. Default -1 = one Hill radius, a (q/3)^(1/3); 0 = off.
* `react_M1`     : if True the primary feels the dark matter. Default False.

Both are written to the output file (`scheme`, `r_partition2`, `react_M1` attributes of `data`).

## Why

1. Lag: with the long step taken first and the inner particles frozen, the inner cusp lagged M1 by half a step and
   pumped energy into the binary (2e-2 rho a^3 G M1/a per orbit at M2 = 7, as q^2). Here every force of the dark matter
   on a black hole is evaluated with both at the same time.
2. Unresolved passages of the secondary: without short steps near M2, about 1% of the particles per ten orbits at
   M2 = 7 received a huge wrong kick (final speeds of 20 v_orb), ten times the binary's loss in spurious energy.
   With short steps inside the Hill radius the largest single gain drops by a factor ~50.

## Checked (M2 = 7, 10^4 particles, 10 orbits, same initial conditions)

* binary loss 22.8 (r_partition2 = 0) and 24.8 (Hill) against 23.5 for the lag-fix scheme;
* with react_M1: total momentum conserved to 1e-14, total energy to 0.5% of the binary loss (with the Hill region);
* with react_M1 the two-body energy loss doubles: the primary moves through a spike at rest in the centre-of-mass
  frame. Do not switch it on without revisiting the initial conditions.
