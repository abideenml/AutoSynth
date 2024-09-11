import argparse
import yaml


# Step 2: Load and parse the YAML file
with open('test.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(config['datasets'])

print(next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'input'), None))