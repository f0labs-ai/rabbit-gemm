# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 F0 Labs

"""CPU checks of the architecture list: what `SR_CUDA_ARCH` may say and which GPUs a build serves."""

import pytest

from sr_vllm.archs import DEFAULT_ARCHS, SmArch, parse_archs, supporting


def test_parses_a_list_into_gencode_targets() -> None:
    archs = parse_archs("90a, 100,120")
    assert archs == (SmArch(9, 0, True), SmArch(10, 0, False), SmArch(12, 0, False))
    assert [a.name for a in archs] == ["90a", "100", "120"]
    assert [a.capability for a in archs] == [90, 100, 120]
    assert archs[0].gencode == "-gencode=arch=compute_90a,code=sm_90a"
    assert archs[2].gencode == "-gencode=arch=compute_120,code=sm_120"


@pytest.mark.parametrize("text", ["", "90a,", "sm_90", "9", "90b", "90a,90a", "90a;120", "1200a"])
def test_rejects_what_nvcc_would_not_take(text: str) -> None:
    with pytest.raises(ValueError, match="SM architecture"):
        parse_archs(text)


def test_default_build_covers_the_measured_gpus() -> None:
    archs = parse_archs(DEFAULT_ARCHS)
    # H100 / H200 are sm_90; RTX 5090 and RTX PRO 6000 Blackwell are sm_120.
    assert supporting(archs, 9, 0) == SmArch(9, 0, True)
    assert supporting(archs, 12, 0) == SmArch(12, 0, False)
    # B200 (sm_100), RTX 4090 / L40S (sm_89) and A100 (sm_80) have no code in the default build.
    assert supporting(archs, 10, 0) is None
    assert supporting(archs, 8, 9) is None
    assert supporting(archs, 8, 0) is None


def test_an_exact_variant_runs_only_on_its_architecture() -> None:
    exact, family = SmArch(10, 0, True), SmArch(10, 0, False)
    assert exact.runs_on(10, 0) and not exact.runs_on(10, 3)
    # Plain sm_100 code is forward compatible within the major: it runs on sm_103 (B300).
    assert family.runs_on(10, 0) and family.runs_on(10, 3)
    assert not family.runs_on(12, 0) and not family.runs_on(9, 0)
    # With both built, a GPU runs the closest one.
    assert supporting((SmArch(12, 0, False), SmArch(12, 1, False)), 12, 1) == SmArch(12, 1, False)
    assert supporting((SmArch(12, 0, False), SmArch(12, 1, False)), 12, 0) == SmArch(12, 0, False)
