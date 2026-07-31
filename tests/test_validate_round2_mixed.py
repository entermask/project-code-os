import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def load_validator() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "canary"
        / "higgs-sweetspot"
        / "benchmark"
        / "validate_round2_mixed.py"
    )
    spec = importlib.util.spec_from_file_location("validate_round2_mixed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def result_config(marker: Any = "R0000007", top_k: Any = 1) -> dict[str, Any]:
    return {
        "run_marker": marker,
        "sampling": {
            "top_k": top_k,
        },
    }


def test_require_paired_config_accepts_matching_fixed_marker_and_top_k() -> None:
    long_result = {"config": result_config()}
    short_result = {"config": result_config()}

    assert VALIDATOR.require_paired_config(long_result, short_result) == (
        "R0000007",
        1,
    )


def test_require_paired_config_accepts_paired_omitted_top_k() -> None:
    long_result = {"config": result_config(top_k=None)}
    short_result = {"config": result_config(top_k=None)}

    assert VALIDATOR.require_paired_config(long_result, short_result) == (
        "R0000007",
        None,
    )


def test_require_run_identity_binds_marker_to_absolute_run_index() -> None:
    long_result = {"label": "dual-mixed-long-r7"}
    short_result = {"label": "dual-mixed-short-r7"}

    assert (
        VALIDATOR.require_run_identity(
            long_result,
            short_result,
            "R0000007",
        )
        == 7
    )


@pytest.mark.parametrize(
    ("long_label", "short_label", "marker"),
    [
        ("dual-mixed-long-r7", "dual-mixed-short-r8", "R0000007"),
        ("dual-mixed-long-r7", "dual-mixed-short-r7", "R0000008"),
        ("dual-long-r7", "dual-mixed-short-r7", "R0000007"),
    ],
)
def test_require_run_identity_rejects_unpaired_labels_or_marker(
    long_label: str, short_label: str, marker: str
) -> None:
    with pytest.raises(AssertionError):
        VALIDATOR.require_run_identity(
            {"label": long_label},
            {"label": short_label},
            marker,
        )


@pytest.mark.parametrize(
    ("long_marker", "short_marker"),
    [
        ("R000007", "R000007"),
        ("R00000007", "R00000007"),
        ("R0000007", "S0000007"),
        ("R00000é7", "R00000é7"),
    ],
)
def test_require_paired_config_rejects_invalid_or_mismatched_markers(
    long_marker: str, short_marker: str
) -> None:
    long_result = {"config": result_config(marker=long_marker)}
    short_result = {"config": result_config(marker=short_marker)}

    with pytest.raises(AssertionError):
        VALIDATOR.require_paired_config(long_result, short_result)


@pytest.mark.parametrize(
    ("long_top_k", "short_top_k"),
    [(0, 0), (1, 2), ("1", "1"), (None, 1), (1, None), (True, True)],
)
def test_require_paired_config_rejects_invalid_or_mismatched_top_k(
    long_top_k: Any, short_top_k: Any
) -> None:
    long_result = {"config": result_config(top_k=long_top_k)}
    short_result = {"config": result_config(top_k=short_top_k)}

    with pytest.raises(AssertionError):
        VALIDATOR.require_paired_config(long_result, short_result)
