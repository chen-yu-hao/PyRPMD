"""Short full-dimensional integration check; does not measure a reaction rate."""
from common import metadynamics


def build():
    return metadynamics(production=False)
