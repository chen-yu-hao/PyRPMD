# TS-only teaching example

`pyrpmd prepare examples/rate/pilot.inp --pilot` evaluates the analytic PES,
forms a finite-difference mass-weighted Hessian, follows its unique unstable
mode in both directions, and identifies the equivalent A + BC / AB + C
channels. `pyrpmd run examples/rate/pilot.inp` runs umbrella sampling, the
flux-metric PMF, constrained parent chains, paired recrossing trajectories,
and parent-cluster/block uncertainty estimates. `minimal.inp` demonstrates
the input containing only the TS structure and PES selection.

`toy_exchange.py` is a newly written analytic teaching potential with Morse
reactant asymptotes. The H labels supply convenient isotope masses and
connectivity radii; **these are not fitted H3 parameters or a physical H + H2
rate benchmark**. The pilot durations are deliberately short. A successful
pipeline run does not establish bead, timestep, Rinf, sampling, or plateau
convergence. The output reports that distinction explicitly.

The mathematical rate definition follows RPMDrate: umbrella forces include
the flux metric, a constrained canonical parent samples the dividing surface,
and unconstrained Maxwell momenta generate paired ±p children. A reported
`rate_estimate_cm3_s` is a finite-time estimate; `rate_cm3_s` is withheld when
overlap, the chosen time-plateau check, or endpoint energy-drift check fails.
No unsuccessful child trajectory is replaced with an assumed transmission.
