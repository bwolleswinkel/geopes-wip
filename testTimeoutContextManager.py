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

	Parameters
	----------
	seconds : float | None
		Number of seconds before timing out. If `None` or non-positive, no timeout is set.

	Notes
	-----
	Uses Unix signals, so it only works in the main thread on POSIX systems.
	
	"""
	# FIXME: Maybe remove the negative seconds is no timeout, seems a bit odd...
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
		with raise_timeout(5):
			for _ in range(1_000_000_000):
				pass
		print("Completed (FAILURE: This should have timed out)")
	except ComputationTimeoutError as e:
		print("Caught timeout (expected):", e)

	print("Demo 2: should finish")
	try:
		with raise_timeout(5):
			time.sleep(3)
		print("Completed (expected)")
	except ComputationTimeoutError as e:
		print("FAILURE:", e)

	print("Demo 3: no timeout set")
	try:
		with raise_timeout(None):
			time.sleep(6)
		print("Completed (expected)")
	except ComputationTimeoutError as e:
		print("FAILURE:", e)

