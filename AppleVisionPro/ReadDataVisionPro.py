import ijson
import numpy as np

# Open the JSON file for incremental reading
with open('streamed_data.json', 'r') as f:
    # Use ijson to parse the JSON file incrementally
    parser = ijson.items(f, 'item')  # 'item' assumes the JSON file contains a top-level list

    for i, frame in enumerate(parser):
        # Extract the keys (names) and shapes from the first frame
        if i == 0:
            names = frame.keys()
            shapes = {key: np.array(frame[key]).shape for key in names}
            print("Data Keys (Names):", list(names))
            print("\nShapes of Data:")
            for key, shape in shapes.items():
                print(f"  {key}: {shape}")
            print("\nFirst Frame:")
            for key, value in frame.items():
                print(f"  {key}:\n{np.array(value)}")

        # Stop after processing the first frame
        break
