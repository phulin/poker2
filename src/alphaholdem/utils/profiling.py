"""
Optional profiling utilities.

This module provides optional profiling decorators that gracefully degrade
when line_profiler is not available.
"""

try:
    from line_profiler import profile

    PROFILE_AVAILABLE = True
except ImportError:
    # Create a no-op decorator when line_profiler is not available
    def profile(func):
        """No-op decorator when line_profiler is not available."""
        return func

    PROFILE_AVAILABLE = False

# Export the profile decorator (either real or no-op)
__all__ = ["profile", "PROFILE_AVAILABLE"]
