import time
import numpy as np
import signal
import sys
from multiprocessing import shared_memory
from avp_stream import VisionProStreamer

avp_ip = "10.31.158.93"

s = VisionProStreamer(ip=avp_ip, record=True)


# Define shared memory parameters
SHM_NAME = "SharedMemory_AVP"
#SHM_SIZE = 8 + 3* 16 * 8  # 8 bytes for Ready flag + 6 * 16 doubles (4x4 matrix)
SHM_SIZE = 8 + 1* 16 * 8  # 8 bytes for Ready flag + 6 * 16 doubles (4x4 matrix)

# Create shared memory
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
    print("Python created shared memory.")
except FileExistsError:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
    print("Python attached to existing shared memory.")

# Define shared memory regions
version = np.ndarray((1,), dtype=np.int64, buffer=shm.buf[:8])  # Ready flag
data = np.ndarray((1, 4, 4), dtype=np.float64, buffer=shm.buf[8:])  # 4x4 matrix

# Initialization
version[0] = -1  # Indicate that Python is initializing
print("Python initialized shared memory. Waiting for C++ to start...")

# Set flag to 0 to signal readiness
time.sleep(1)  # Simulate initialization delay
version[0] = 0
print("Python is ready. Ready flag set to 0.")

try:
    while True:
        r = s.latest  # Get the latest data

        # Collect all data into a single numpy array (data_storage)
        data_storage = np.array([
            r['right_wrist'][0],               # data_25
            #r['right_fingers'][0],         # data_25
            #r['right_fingers'][10],         # data_10
            #r['right_fingers'][11],         # data_11
            #r['right_fingers'][12],         # data_12
            #r['right_fingers'][13],         # data_13
            #r['right_fingers'][14],         # data_14
        ], dtype=np.float64)

        # Write data to shared memory only if Ready flag is 0
        if version[0] == 0:
            data[:] = data_storage 
            version[0] = 1  # Set Ready flag to 1


except KeyboardInterrupt:
    print("Python: Stopping.")

finally:
    shm.close()
    shm.unlink()
    print("Shared memory cleaned up.")