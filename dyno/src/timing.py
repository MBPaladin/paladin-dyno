import ctypes

# This function provides hooks into clock_nanosleep, a higher precision, non blocking, timing function
# If a higher level of timing synchronization is required a busy wait loop can be substituted

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
    
# blocking sleep function
# def ns_delay(target):
#     while time.perf_counter_ns() < target:
#         pass