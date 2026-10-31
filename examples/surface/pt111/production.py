"""600 K / 40 ns / three-replica settings. Read README.md before production."""
from common import metadynamics


def build():
    return metadynamics(production=True)
