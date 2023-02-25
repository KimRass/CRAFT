import argparse
from time import time
from datetime import timedelta


def get_arguments():
    parser = argparse.ArgumentParser(description="train_craft")

    parser.add_argument("--data_dir")

    args = parser.parse_args()
    return args


def get_elapsed_time(time_start):
    return timedelta(seconds=round(time() - time_start))
