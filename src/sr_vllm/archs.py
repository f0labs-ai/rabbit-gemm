# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 F0 Labs

"""SM architectures the compiled extension carries, and which GPUs they run on.

A CUDA binary holds machine code per SM architecture. Code built for `sm_XY` runs on GPUs of the same major
architecture whose minor is at least Y. Code built for an `a` variant (`90a`) may use instructions of exactly
that architecture, so it is run only there.

This module imports nothing heavy: `setup.py` uses it to turn `SR_CUDA_ARCH` into nvcc flags, and the plugin
uses it to refuse a GPU the binary has no code for before vLLM loads any weights.
"""

import re
from dataclasses import dataclass

# What the extension is built for when SR_CUDA_ARCH is unset: every architecture the kernels have been
# measured on. 90a: H100 / H200. 120: RTX 5090 and RTX PRO 6000 Blackwell.
DEFAULT_ARCHS = "90a,120"

_ARCH = re.compile(r"(\d{1,2})(\d)(a?)")


@dataclass(frozen=True)
class SmArch:
    """One `-gencode` target, e.g. 90a or 120."""

    major: int
    minor: int
    exact: bool  # an `a` variant: runs on exactly this architecture

    @property
    def name(self) -> str:
        return f"{self.major}{self.minor}{'a' if self.exact else ''}"

    @property
    def capability(self) -> int:
        """vLLM's capability number: 10 x major + minor."""
        return self.major * 10 + self.minor

    @property
    def gencode(self) -> str:
        return f"-gencode=arch=compute_{self.name},code=sm_{self.name}"

    def runs_on(self, major: int, minor: int) -> bool:
        if major != self.major:
            return False
        return minor == self.minor if self.exact else minor >= self.minor


def parse_archs(text: str) -> tuple[SmArch, ...]:
    """Parses a comma-separated list such as "90a,120". Raises ValueError on anything else."""
    archs: list[SmArch] = []
    for raw in text.split(","):
        token = raw.strip()
        match = _ARCH.fullmatch(token)
        if match is None:
            raise ValueError(f"SM architecture must look like 90a or 120, got {token!r} in {text!r}")
        arch = SmArch(int(match[1]), int(match[2]), match[3] == "a")
        if arch in archs:
            raise ValueError(f"SM architecture {arch.name} is listed twice in {text!r}")
        archs.append(arch)
    return tuple(archs)


def supporting(archs: tuple[SmArch, ...], major: int, minor: int) -> SmArch | None:
    """The built architecture a GPU of capability (major, minor) runs, or None when the binary has none."""
    usable = [a for a in archs if a.runs_on(major, minor)]
    # The closest one: code for a later minor uses more of the hardware.
    return max(usable, key=lambda a: a.minor, default=None)
