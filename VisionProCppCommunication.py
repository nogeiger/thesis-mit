import time
import numpy as np
from multiprocessing import shared_memory
from avp_stream import VisionProStreamer

# Define shared memory parameters
SHM_NAME = "SharedMemory_AVP"
SHM_SIZE = 8 + 16 * 8  # 8 bytes for version counter + 16 doubles (4x4 matrix)

# Connect to the AVP streamer
avp_ip = "10.31.190.119"  # Replace with your actual IP
s = VisionProStreamer(ip=avp_ip, record=True)

# Create shared memory
shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)

# Define shared memory regions
version = np.ndarray((1,), dtype=np.int64, buffer=shm.buf[:8])  # Version counter
data = np.ndarray((4, 4), dtype=np.float64, buffer=shm.buf[8:])  # 4x4 matrix

# Time settings
timestep = 0.005  # Desired time interval (5 ms)

try:
    while True:
        # Get the latest data from the VisionProStreamer
        r = s.latest  # This contains the transformation matrix (or relevant data)

        # Extract the 'right_wrist' transformation matrix
        right_wrist = np.array(r['right_wrist'], dtype=np.float64)  # Ensure it's a NumPy array
        if right_wrist.shape != (4, 4):
            raise ValueError("Expected a 4x4 matrix for 'right_wrist'.")

        # Increment the version counter (start)
        version[0] += 1

        # Write the matrix to shared memory
        data[:] = right_wrist

        # Increment the version counter (end)
        version[0] += 1

        # Sleep for the desired timestep
        time.sleep(timestep)

except KeyboardInterrupt:
    print("Data streaming stopped.")

finally:
    # Clean up shared memory
    shm.close()
    shm.unlink()
