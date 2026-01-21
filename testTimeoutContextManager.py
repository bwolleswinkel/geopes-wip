"""Script to generate a contextmanager which times out after a given duration"""

# FROM: GitHub Copilot GPT-5 mini | 2026/01/21

import signal
from contextlib import contextmanager
from socket import timeout
import threading
import time


class ComputationTimeoutError(TimeoutError):
	"""Raised when a computation times out. Should be used used with the context manager `raise_timeout`."""
	pass


@contextmanager
def raise_timeout(seconds: float | None):
	"""Context manager that raises `ComputationTimeoutError` if the block exceeds `seconds`.

	Usage:
		with raise_timeout(2):
			long_running_call()

	Notes:
	- Uses Unix signals, so it only works in the main thread on POSIX systems.
	
	"""
	if seconds is None or seconds <= 0:
		yield
		return
	if not isinstance(seconds, (int, float)):
		raise TypeError("Seconds must be int or float")
	if threading.current_thread() is not threading.main_thread():
		raise RuntimeError("raise_timeout contextmanager only works in main thread (uses signals)")

	def _handler(signum, frame):
		raise ComputationTimeoutError(f"Operation timed out after {seconds} seconds")

	old_handler = signal.getsignal(signal.SIGALRM)
	signal.signal(signal.SIGALRM, _handler)
	signal.setitimer(signal.ITIMER_REAL, seconds)
	try:
		yield
	finally:
		signal.setitimer(signal.ITIMER_REAL, 0)
		signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
	print("Demo 1: should timeout")
	try:
		with raise_timeout(3):
			for _ in range(1_000_000_000):
				pass
		print("Completed (FAILURE: This should have timed out)")
	except ComputationTimeoutError as e:
		print("Caught timeout:", e)

	print("Demo 2: should finish")
	try:
		with raise_timeout(3):
			time.sleep(1)
		print("Completed (expected)")
	except ComputationTimeoutError as e:
		print("Unexpected timeout:", e)

