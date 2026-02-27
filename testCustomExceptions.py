"""Script to test generating custom exceptions for the NumPES library"""


class NumpesException(Exception):
    """Base exception class for the NumPES library"""
    pass


class DimensionsError(NumpesException, ValueError):
    """Exception raised when there is a mismatch in dimensions"""
    pass


class ConsistencyError(NumpesException, AssertionError):
    """Exception raised when an internal inconsistency is detected"""
    pass


class InfeasibilityError(NumpesException, RuntimeError):
    """Exception raised when an optimization problem is detected to be infeasible"""
    pass


class TimeLimitExceeded(NumpesException, TimeoutError):
    """Exception raised when a time limit is exceeded during calculation"""
    pass


class Dummy:

    def raise_dimensions_error(self):
        raise DimensionsError("This is a test of the DimensionsError exception")
    
    def raise_consistency_error(self):
        raise ConsistencyError("This is a test of the ConsistencyError exception")
    
    def raise_infeasibility_error(self):
        raise InfeasibilityError("This is a test of the InfeasibilityError exception")
    
    def raise_time_limit_exceeded(self):
        raise TimeLimitExceeded("This is a test of the TimeLimitExceeded exception")


def main():
    """Main function to run tests for custom exceptions"""
    
    raiser = Dummy()

    try:
        raiser.raise_dimensions_error()
    except DimensionsError as e:
        print(f"Caught DimensionsError: {e}")

    try:
        raiser.raise_dimensions_error()
    except ValueError as e:
        print(f"Caught ValueError: {e}")


if __name__ == "__main__":
    main()