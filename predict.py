import argparse
from ultralytics import YOLO

# Command-line argument parser for image path
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help='Path to the image file')
args = parser.parse_args()

# Load the trained model
checkpoint_path = "/root/.cache/labels7/weights.pt"
model = YOLO(checkpoint_path)

# Make predictions on the provided image
results = model(args.image)

# Display the results
results.show()  # Optionally, save results using results.save() if required
