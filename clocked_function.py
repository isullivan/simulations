import time


def clocked_function(func):
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print('Timing for ' + func.__name__ +': [%0.8fs]' % (elapsed))
        return result
    return clocked
