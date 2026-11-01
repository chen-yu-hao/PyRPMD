"""An analytic two-particle OpenMM teaching model, not a fitted H2 PES."""


def build_system():
    import openmm

    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = openmm.HarmonicBondForce()
    force.addBond(0, 1, 0.074, 300000.0)  # nm, kJ mol^-1 nm^-2
    system.addForce(force)
    return system
