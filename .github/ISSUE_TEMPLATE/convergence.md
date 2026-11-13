---
name: Calculation or convergence question
about: Diagnose sampling, PMF support, rates, diffusion or spectra
title: ""
labels: ""
assignees: ""
---

**Observable, model and ensemble**
State the target quantity, PES/model version, temperature, masses/isotopes,
bead count, timestep and sampling ensemble.

**Current result**
Include the resolved input and summary, especially status, ready/converged flags,
failed or unresolved histories, and uncertainty availability.

**Sampling evidence**
Give equilibration/production budgets and independent replicas. For PMFs include
overlap, ESS and unsupported bins; for rates include parent/child budgets and
the transmission plateau; for diffusion/spectra include the fitted time window
or record length and frequency resolution.

**Changes already compared**
List timestep, bead, budget or domain changes and their observed effect. Follow
the repository's `docs/convergence.md` guide. Do not discard failed histories or
retry random seeds until a desired result appears.
