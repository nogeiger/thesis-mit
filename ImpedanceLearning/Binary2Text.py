import numpy as np
import pandas as pd

# File paths
binary_file = "File_data.bin"  # Update with correct path
text_file = "File_data.txt"  # Output readable text file

# Define the correct number of categories (columns)
column_count = 13  # We confirmed earlier

# Define the column names based on C++ variables
column_names = [
    "time", "f_x", "f_y", "f_z", 
    "m_x", "m_y", "m_z", 
    "x", "y", "z", 
    "x0", "y0", "z0"
]

# Read the binary file as double-precision floats (float64)
data = np.fromfile(binary_file, dtype=np.float64)

# Reshape data into the correct structure
if len(data) % column_count != 0:
    print("Error: Data length is not a multiple of 13. Check binary structure.")
    exit()

data = data.reshape(-1, column_count)  # Reshape into rows with 13 columns

# Convert to Pandas DataFrame with column names
df = pd.DataFrame(data, columns=column_names)

# Print a preview of the data
print(df.head())  # Shows the first few rows

# Save to a readable text file (CSV format, tab-separated)
df.to_csv(text_file, sep="\t", index=False)

print(f"Binary file successfully converted to {text_file} with column names.")
