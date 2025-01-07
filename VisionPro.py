import json

from avp_stream import VisionProStreamer

avp_ip = "10.31.190.119"  # Replace with your actual IP
s = VisionProStreamer(ip=avp_ip, record=True)

data = []  # List to store the data
try:
    while True:
        r = s.latest
        # Append the data you want to store
        data.append({
            'head': r['head'].tolist(),  # Convert numpy arrays to lists
            'right_wrist': r['right_wrist'].tolist(),
            'right_fingers': r['right_fingers'].tolist()
        })
except KeyboardInterrupt:
    # Save to a JSON file when exiting
    with open('streamed_data.json', 'w') as f:
        json.dump(data, f)
    print("Data saved to streamed_data.json")
