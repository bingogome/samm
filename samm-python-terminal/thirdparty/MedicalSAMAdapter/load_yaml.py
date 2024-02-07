import yaml
import os
import argparse


def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_argparse_config(config):
    parser = argparse.ArgumentParser()

    # Loop through each key-value pair in the config and add them as arguments
    for key, value in config.items():
        if type(value) == bool:
            # For boolean values, need to use 'store_true' or 'store_false'
            parser.add_argument(
                f"--{key}", action="store_true" if not value else "store_false"
            )
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    return parser.parse_args([])  # parse_args with an empty list to use defaults


def get_argparse_config():
    config_path = os.path.join(os.path.dirname(__file__), "../../config/training.yaml")
    config = load_config(config_path)
    return create_argparse_config(config)


# Convert the YAML configuration to argparse
