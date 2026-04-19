"""
TSTA Advanced Analysis Package
================================
9-phase deep experimental validation of trajectory representation properties.
"""
from tsta_project.analysis.direction_invariance  import run_direction_invariance
from tsta_project.analysis.amplitude_invariance  import run_amplitude_invariance
from tsta_project.analysis.domain_shift          import run_domain_shift
from tsta_project.analysis.temporal_smoothness   import run_temporal_smoothness
from tsta_project.analysis.geometric_structure   import run_geometric_structure
from tsta_project.analysis.time_reversal         import run_time_reversal
from tsta_project.analysis.partial_signal        import run_partial_signal
from tsta_project.analysis.failure_modes         import run_failure_modes
from tsta_project.analysis.research_report       import print_research_report

__all__ = [
    "run_direction_invariance",
    "run_amplitude_invariance",
    "run_domain_shift",
    "run_temporal_smoothness",
    "run_geometric_structure",
    "run_time_reversal",
    "run_partial_signal",
    "run_failure_modes",
    "print_research_report",
]
