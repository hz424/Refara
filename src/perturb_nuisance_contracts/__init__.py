"""Shared contract utilities that do not import benchmark modules."""


class BenchmarkContractError(ValueError):
    """Raised when inputs would make a benchmark comparison invalid."""


__all__ = ["BenchmarkContractError"]
