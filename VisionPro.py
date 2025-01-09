import time
from avp_stream import VisionProStreamer

avp_ip = "192.168.24.10"  # Replace with your actual IP
s = VisionProStreamer(ip=avp_ip, record=True)

# Open a text file for writing
with open('streamed_data.txt', 'w') as f:
    start_time = time.time()  # Capture the start time
    try:
        while True:
            r = s.latest
            #print("Raw data from streamer:", r)
            # Calculate relative timestamp, so that it starts from 0
            relative_timestamp = time.time() - start_time

            # Write data to the text file
            f.write(f"Timestamp: {relative_timestamp:.6f}\n")
            f.write(f"Head: {r['head'].tolist()}\n")
            f.write(f"Right Wrist: {r['right_wrist'].tolist()}\n")
            #f.write(f"Left Wrist: {r['left_wrist'].tolist()}\n")
            #f.write(f"Right Fingers: {r['right_fingers'].tolist()}\n")
            #f.write(f"Left Fingers: {r['left_fingers'].tolist()}\n")
            #f.write(f"Right Pinch Distance: {r['right_pinch_distance']:.6f}\n")
            #f.write(f"Left Pinch Distance: {r['left_pinch_distance']:.6f}\n")
            #f.write(f"Right Wrist Roll: {r['right_wrist_roll']:.6f}\n")
            #f.write(f"Left Wrist Roll: {r['left_wrist_roll']:.6f}\n")
            f.write("\n")  # Separate each frame with a blank line
            break
            # Flush to ensure data is written immediately
            f.flush()
    except KeyboardInterrupt:
        print("Data streaming stopped. Saved to streamed_data.txt")
