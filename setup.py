"""Minimal build: Python C API avoids a NumPy ABI/build-header dependency."""

import os
import shlex
import shutil
import sys
import sysconfig
from setuptools import Extension, setup

extensions = []
if os.environ.get("PYRPMD_NO_NATIVE", "0") != "1":
    # Some portable Python builds remember a compiler which is absent on the
    # runtime host. Respect explicit choices, otherwise choose an available one.
    for variable, candidates in (
        ("CC", ("cc", "gcc", "clang")),
        ("CXX", ("c++", "g++", "clang++")),
    ):
        configured = shlex.split(sysconfig.get_config_var(variable) or "")
        if variable not in os.environ and (
            not configured or shutil.which(configured[0]) is None
        ):
            replacement = next(
                (shutil.which(name) for name in candidates if shutil.which(name)), None
            )
            if replacement:
                os.environ[variable] = replacement
    compile_args = (
        ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"]
    )
    link_args = []
    if (
        sys.platform.startswith("linux")
        and os.environ.get("PYRPMD_NO_OPENMP", "0") != "1"
    ):
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")
    extensions.append(
        Extension(
            "pyrpmd._native",
            ["native/cayley.cpp"],
            depends=["native/propagation.h", "native/include/pyrpmd/pes_api_v1.h"],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    )
setup(
    ext_modules=extensions,
    license_files=["LICENSE", "THIRD_PARTY_NOTICES.md"],
    data_files=[
        ("include/pyrpmd", ["native/include/pyrpmd/pes_api_v1.h"]),
        ("share/pyrpmd/examples", ["native/examples/harmonic_pes.c"]),
    ],
)
