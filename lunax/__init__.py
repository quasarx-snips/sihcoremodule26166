"""Stable package facade for the LunaX registration pipeline."""

from .pipeline import (
    PipelineConfig,
    RegistrationResult,
    run_lunax_from_arrays,
    run_lunax_registration,
    print_lunax_pipeline_report,
)

__all__ = [
    "PipelineConfig", "RegistrationResult", "run_lunax_from_arrays",
    "run_lunax_registration",
    "print_lunax_pipeline_report",
]
