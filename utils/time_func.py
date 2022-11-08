import math
import time

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def timeit(logger):
    def log(func):
        def wrapped(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            # logger.info(f'** Function {func.__name__}{args} Took {asMinutes(total_time)}')
            logger.info(f'** Function {func.__name__} Took {asMinutes(total_time)}')
            #{total_time: .4f}
            return result

        return wrapped
    return log