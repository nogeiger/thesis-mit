import time
from avp_stream import VisionProStreamer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

<<<<<<< Updated upstream
avp_ip = "10.29.171.74"  # Replace with your actual IP
=======

avp_ip = "10.29.158.147"  # Replace with your actual IP
>>>>>>> Stashed changes
s = VisionProStreamer(ip=avp_ip, record=True)

# Placeholder for translational data
translations = []
i=0

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
                f.write(f"Right Finger Knuckle: {r['right_fingers'][11].tolist()}\n")
                f.write(f"Right Finger intermediate base: {r['right_fingers'][12].tolist()}\n")
                f.write(f"Right Finger tip: {r['right_fingers'][14].tolist()}\n")
                f.write("\n")  # Separate each frame with a blank line

                # Flush to ensure data is written immediately
                f.flush()

                # Update the next timestep
                next_timestep += timestep

                #for key in r.keys():
                #    print(key)
                #print("right_fingers 0 ", r['right_fingers'][0])
                #print("right_fingers len ", len(r['right_fingers']))
                #print("right_fingers 25", r['right_fingers'][24])
                #print("right_wrist", r['right_wrist'])

                # Get the right_wrist matrix
                #right_wrist = r['right_wrist']

                #print(r['right_fingers'][11])



                matrix = r['right_fingers'][1]
                    
                # Extract the translational component (last column excluding the 1.0)
                translation = [matrix[0][3], matrix[1][3], matrix[2][3]]
                    
                # Append the translation to the list
                translations.append((i, translation))

                i=+1
                if i ==250:
                    break

                



                

            # Sleep for a short duration to avoid busy waiting
            time.sleep(0.001)  # 1 ms sleep to reduce CPU usage

    except KeyboardInterrupt:
        print("Data streaming stopped. Saved to streamed_data.txt")



    # Separate data for plotting
    ids = [t[0] for t in translations]  # Index IDs
    x = [t[1][0] for t in translations]  # X translations
    y = [t[1][1] for t in translations]  # Y translations
    z = [t[1][2] for t in translations]  # Z translations

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points and add text labels for IDs
    ax.scatter(x, y, z, c='blue', marker='o', label="Translation Points")


                # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Translation Positions with IDs')

    plt.legend()
    plt.show()