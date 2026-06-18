#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
from model import PFKDE

from dataclasses import dataclass



@dataclass
class CustomParameters:
    n_bins: int = 5782
    omission_threshold: float = 8*10**-7
    n_minimum_points: int = 15
    aggregation_window_size: int = 15
    memory_size: int = 300
    bandwidth_function: int = 1
    weight_function: int = 1
    threshold_type: int = -1



class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def load_data(config: AlgorithmArgs) -> np.ndarray:
    df = pd.read_csv(config.dataInput)
    data = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    contamination = labels.sum() / len(labels)
    # Use smallest positive float as contamination if there are no anomalies in dataset
    contamination = np.nextafter(0, 1) if contamination == 0. else contamination
    return data, contamination


def main(config: AlgorithmArgs):
    data, contamination = load_data(config)

    model = PFKDE(
        n_bins=config.customParameters.n_bins,
        omission_threshold=config.customParameters.omission_threshold,
        n_minimum_points=config.customParameters.n_minimum_points,
        aggregation_window_size=config.customParameters.aggregation_window_size,
        memory_size=config.customParameters.memory_size,
        bandwidth_function=config.customParameters.bandwidth_function,
        weight_function=config.customParameters.weight_function,
        threshold_type=config.customParameters.threshold_type,
    )

    model.fit(data)
    scores = model.decision_scores_
    np.savetxt(config.dataOutput, scores, delimiter=",")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()

    if config.executionType == "train":
        print("Nothing to train, finished!")
    elif config.executionType == "execute":
        main(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")
