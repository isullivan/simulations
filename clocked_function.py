import time


def clocked_function(func):
    def clocked(*args):
        t0 = time.time()
        result = func(*args)
        elapsed = time.time() - t0
        print('Timing: [%0.8fs]' % (elapsed))
        return result
    return clocked
