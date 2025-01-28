import time
import numpy as np
from multiprocessing import shared_memory
from avp_stream import VisionProStreamer

# Define shared memory parameters
SHM_NAME = "SharedMemory_AVP"
SHM_SIZE = 16 * 8 + 8  # 16 doubles (4x4 matrix) + 1 int64 (ready flag)

# Connect to the AVP streamer
avp_ip = "10.31.190.119"  # Replace with your actual IP
s = VisionProStreamer(ip=avp_ip, record=True)

# Create shared memory
shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)

# Define shared memory regions
data = np.ndarray((4, 4), dtype=np.float64, buffer=shm.buf[:16 * 8])  # 4x4 matrix
ready_flag = np.ndarray((1,), dtype=np.int64, buffer=shm.buf[16 * 8:])  # Single int64

# Time settings
timestep = 0.005  # Desired time interval (5 ms)
start_time = time.time()
next_timestep = timestep

try:
    while True:
        current_time = time.time()
        relative_timestamp = current_time - start_time

        # Check if it's time to update the shared memory
        if relative_timestamp >= next_timestep:
            # Get the latest data from the VisionProStreamer
            r = s.latest  # This contains the transformation matrix (or relevant data)

            # Extract the 'right_wrist' transformation matrix
            right_wrist = np.array(r['right_wrist'], dtype=np.float64)  # Ensure it's a NumPy array
            if right_wrist.shape != (4, 4):
                raise ValueError("Expected a 4x4 matrix for 'right_wrist'.")

            # Write the matrix to shared memory
            data[:] = right_wrist

            # Set the ready flag to 1 (indicating data is ready for C++)
            ready_flag[0] = 1

            # Update the next timestep
            next_timestep += timestep

        # Sleep for a short duration to avoid busy waiting
        time.sleep(0.001)  # 1 ms sleep to reduce CPU usage

except KeyboardInterrupt:
    print("Data streaming stopped.")

finally:
    # Clean up shared memory
    shm.close()
    shm.unlink()
