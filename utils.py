import time
from functools import wraps

def timeit(func):
	@wraps(func)
	def time_func(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		elapsed = time.time() - start_time
		print("[TIMEIT] %s --- %fs" % (func.__name__, elapsed))
		return result
	return time_func