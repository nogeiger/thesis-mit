import time
from avp_stream import VisionProStreamer

avp_ip = "10.31.169.22"  # Replace with your actual IP
s = VisionProStreamer(ip=avp_ip, record=True)

# Open a text file for writing
with open('streamed_data.txt', 'w') as f:
    start_time = time.time()  # Capture the start time
    last_written_time = 0  # Initialize the last written timestamp
    timestep = 0.005  # Desired time interval (5 ms)
    next_timestep = timestep  # Start with the first timestep

    try:
        while True:
            current_time = time.time()
            relative_timestamp = current_time - start_time

            # Check if the current relative timestamp has reached or exceeded the next timestep
            if relative_timestamp >= next_timestep:
                r = s.latest  # Get the latest data

                # Write data to the text file
                f.write(f"Timestamp: {next_timestep:.6f}\n")
                f.write(f"Right Wrist: {r['right_wrist'].tolist()}\n")
                f.write(f"Right Wrist roll: {r['right_wrist_roll'].tolist()}\n")

                f.write("\n")  # Separate each frame with a blank line

                # Flush to ensure data is written immediately
                f.flush()

                # Update the next timestep
                next_timestep += timestep

            # Sleep for a short duration to avoid busy waiting
            time.sleep(0.001)  # 1 ms sleep to reduce CPU usage

    except KeyboardInterrupt:
        print("Data streaming stopped. Saved to streamed_data.txt")