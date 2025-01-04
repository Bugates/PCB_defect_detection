import os
from ultralytics import YOLO
import yaml

# Path to dataset and YAML configuration file
root_dir = "/root/.cache/labels7/"
data_yaml = {
    'path': str(root_dir),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': len(class_dict),
    'names': list(class_dict.values())
}

# Save YAML file
with open(os.path.join(root_dir, 'data.yaml'), 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

# Load YOLO model and train
model = YOLO('yolov8n.pt')
results = model.train(data=os.path.join(root_dir, 'data.yaml'), epochs=1000, patience=0)

# Save model weights
weights_path = os.path.join(root_dir, 'weights.pt')
model.save(weights_path)
print(f"Model weights saved successfully at: {weights_path}")
