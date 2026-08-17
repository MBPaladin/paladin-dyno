import ctypes
import os
import time

# This function provides hooks into clock_nanosleep, a higher precision, non blocking, timing function
# If a higher level of timing synchronization is required a busy wait loop can be substituted

CLOCK_MONOTONIC = 1
TIMER_ABSTIME = 1
EINTR = 4

# Define the timespec structure
class timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

# Load the libc library
libc = ctypes.CDLL("libc.so.6", use_errno=True)

clock_nanosleep = libc.clock_nanosleep
clock_nanosleep.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(timespec), ctypes.POINTER(timespec)]
clock_nanosleep.restype = ctypes.c_int

def nano_sleep(delay_ns):
    ts = timespec()
    ts.tv_sec = delay_ns // 1_000_000_000
    ts.tv_nsec = delay_ns % 1_000_000_000
    ret = clock_nanosleep(0, 0, ctypes.pointer(ts), None)
    if ret != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"clock_nanosleep error: {errno}")

def nano_sleep_until(target_ns):
    """Sleep until an absolute CLOCK_MONOTONIC deadline (ns).

    Absolute deadlines don't accumulate the gap between computing a relative
    delay and entering the syscall, so the cycle schedule cannot drift.
    """
    ts = timespec()
    ts.tv_sec = target_ns // 1_000_000_000
    ts.tv_nsec = target_ns % 1_000_000_000
    while True:
        ret = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, ctypes.pointer(ts), None)
        if ret == 0:
            return
        if ret != EINTR:  # clock_nanosleep returns the error instead of setting errno
            raise OSError(ret, f"clock_nanosleep error: {ret}")

def hybrid_sleep_until(target_ns, spin_margin_ns=150_000):
    """Sleep until target_ns minus a margin, then busy-wait to the deadline.

    The kernel wakes a sleeping thread some tens to hundreds of us late
    depending on load; that wake-up slop is the dominant source of cycle
    jitter. Spinning the final spin_margin_ns absorbs it at the cost of
    ~spin_margin/cycle_time of one core.
    """
    coarse = target_ns - spin_margin_ns
    if time.clock_gettime_ns(time.CLOCK_MONOTONIC) < coarse:
        nano_sleep_until(coarse)
    while time.clock_gettime_ns(time.CLOCK_MONOTONIC) < target_ns:
        pass

def set_cyclic_thread_priority(priority=80):
    """Put the calling thread in the SCHED_FIFO real-time class.

    Needs root or CAP_SYS_NICE; returns False (and the loop runs at normal
    priority) otherwise. pid 0 targets the calling thread on Linux.
    """
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        return True
    except PermissionError:
        return False

# blocking sleep function
# def ns_delay(target):
#     while time.perf_counter_ns() < target:
#         pass