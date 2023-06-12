from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.bee_hive import BeeHive
from src.config import Config
from src.ga import fitness


def load_data(suffix: Literal["dataset", "outputs"], *args, **kwargs) -> tuple[NDArray, NDArray]:
    """Load the training and testing data.

    Args:
        suffix (string): Name part of the file from which the data will be loaded.

    Returns:
        tuple[NDArray, NDArray]: Training and testing data.
    """
    train_data: NDArray = np.empty(*args, **kwargs)
    test_data: NDArray = np.empty(*args, **kwargs)

    for fruit in Config.fruits:
        file_name: str = f"{Config.data_path}/{fruit}_{suffix}.npy"
        data = np.load(file_name)
        train_data_length: int = int(np.round(len(data) * Config.train_data_ratio))
        train: list = data[:train_data_length, ...]
        test: list = data[train_data_length:, ...]
        train_data = np.append(train_data, train, axis=0)
        test_data = np.append(test_data, test, axis=0)

    return train_data, test_data


def main() -> None:
    train_data_input, test_data_input = load_data("dataset", (0, 360))
    train_data_output, test_data_output = load_data("outputs", 0, dtype=int)

    hive = BeeHive(
        shape=[train_data_input.shape[1], 150, 60, 4],
        fitness_func=fitness,
        input_train_data=train_data_input,
        output_train_data=train_data_output,
        input_test_data=test_data_input,
        output_test_data=test_data_output,
    )

    hive.run()


if __name__ == "__main__":
    main()
