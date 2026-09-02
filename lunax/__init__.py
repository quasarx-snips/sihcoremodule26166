"""Stable package facade for the LunaX registration pipeline."""

from module_11 import (
    PipelineConfig,
    RegistrationResult,
    run_lunax_from_arrays,
    run_lunax_registration,
)

__all__ = [
    "PipelineConfig", "RegistrationResult", "run_lunax_from_arrays",
    "run_lunax_registration",
]
