import numpy as np

# Replace with the actual path to your file
file_path = 'C:/Users/MSI/Documents/Git/Tech-Visionaries-fake-news-detection/Video Analysis/data/processed/motion_vectors/aagfhgtpmv_mv.npy'

try:
    # Load the motion vector data
    mv_data = np.load(file_path)

    print(f"Successfully loaded {file_path}")
    print(f"Array Shape: {mv_data.shape}")
    print(f"Data Type: {mv_data.dtype}")

    # Check for non-zero elements to confirm it's not an empty file
    non_zero_elements = np.count_nonzero(mv_data)
    total_elements = mv_data.size
    
    print(f"Total elements: {total_elements}")
    print(f"Number of non-zero elements: {non_zero_elements}")

    if non_zero_elements > 0:
        print("\n✅ This file contains relevant, non-zero motion vector data.")
    else:
        print("\n Warning: This file is structured correctly but contains only zeros.")

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")