"""Small, independently written RPMDrate-style input for conversion only.

PES installation is a separate step. These short sampling budgets do not
establish a physical rate. Coordinates follow the upstream H + H2 atom order.
"""

from h3_bkmp2 import get_potential

reactants(atoms=["H", "H", "H"], reactant1Atoms=[1, 2], reactant2Atoms=[3],
          Rinf=(15.8754, "angstrom"))
transitionState(geometry=([[0, 0, -1.757], [0, 0, 0], [0, 0, 1.757]], "bohr"),
                formingBonds=[(2, 3)], breakingBonds=[(1, 2)])
equivalentTransitionState(formingBonds=[(1, 3)], breakingBonds=[(2, 1)])
thermostat("Andersen", samplingTime=(0.0002, "ps"))
windows = [Window(xi=xi, kforce=0.0001*T, trajectories=1,
                  equilibrationTime=(0.0004, "ps"), evolutionTime=(0.002, "ps"))
           for xi in [0.0, 0.5, 1.0]]
conductUmbrellaSampling(dt=(0.0001, "ps"), windows=windows)
computePotentialOfMeanForce(windows=windows, xi_min=0.0, xi_max=1.0, bins=50)
computeRecrossingFactor(dt=(0.0001, "ps"), equilibrationTime=(0.0004, "ps"),
                       childTrajectories=4, childSamplingTime=(0.0002, "ps"),
                       childrenPerSampling=2, childEvolutionTime=(0.0004, "ps"),
                       xi_current=1.0)
computeRateCoefficient()
