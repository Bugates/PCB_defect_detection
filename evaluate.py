from ultralytics import YOLO
import yaml
from prettytable import PrettyTable

# Path to the trained weights and data.yaml file
root_dir = "/root/.cache/labels7/"
checkpoint_path = os.path.join(root_dir, 'weights.pt')

# Load the trained model
model = YOLO(checkpoint_path)

# Evaluate the model
results = model.val(data=os.path.join(root_dir, 'data.yaml'))

# Display evaluation metrics in a table
evaluation_table = PrettyTable()
evaluation_table.field_names = ["Metric", "Value"]
metrics = results.results_dict
for metric, value in metrics.items():
    evaluation_table.add_row([metric, value])

# Print evaluation table
print(evaluation_table)
