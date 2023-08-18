import argparse
from time import time
from datetime import timedelta


def get_args():
    parser = argparse.ArgumentParser(description="train_craft")

    parser.add_argument("--data_dir")
    parser.add_argument("--batch_size", type=int)

    args = parser.parse_args()
    return args


def get_elapsed_time(time_start):
    return timedelta(seconds=round(time() - time_start))
