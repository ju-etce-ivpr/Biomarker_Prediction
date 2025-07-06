import argparse
import yaml
import os

from train import train_model
from test import evaluate_model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Glioma Biomarker Prediction")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint for testing')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == 'train':
        train_model(config)

    elif args.mode == 'test':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be specified in test mode")
        evaluate_model(config, args.checkpoint)

if __name__ == '__main__':
    main()
