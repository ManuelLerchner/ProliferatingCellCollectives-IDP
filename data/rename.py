import os
import glob

# Ask for new base name
new_name = input("Enter the new base name: ")

# Find all files matching the pattern
files = sorted(glob.glob("new.*.jpeg"))

# Rename each file
for f in files:
    # Extract the number part from "new.XXXX.jpeg"
    number = f.split(".")[1]  # e.g., "0000"
    new_filename = f"{new_name}_{number}.jpeg"
    os.rename(f, new_filename)
    print(f"Renamed: {f} -> {new_filename}")

print("Done!")

