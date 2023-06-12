from enum import Enum, member
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class ActivationFunction(Enum):
    @member
    @staticmethod
    def sigmoid(input_data: NDArray) -> NDArray:
        return 1.0 / (1.0 + np.exp(-1 * input_data))

    @member
    @staticmethod
    def relu(input_data: NDArray) -> NDArray:
        return input_data.clip(min=0)


def predict_outputs(
    weights_mat: NDArray,
    data_inputs: NDArray,
    data_outputs: NDArray,
    activation: str = "relu",
) -> tuple[float, NDArray]:
    size: int = data_inputs.shape[0]
    predictions: NDArray = np.zeros(size)
    activation_func: Callable[[NDArray], NDArray] = ActivationFunction[activation].value

    for sample_idx in range(size):
        r1: NDArray = data_inputs[sample_idx, :]
        for curr_weights in weights_mat:
            r1 = np.matmul(r1, curr_weights)
            r1 = activation_func(r1)

        predicted_label = np.where(r1 == np.max(r1))[0][0]
        predictions[sample_idx] = predicted_label

    correct_predictions: int = np.where(predictions == data_outputs)[0].size
    accuracy: float = (correct_predictions / data_outputs.size) * 100
    return accuracy, predictions


def fitness(
    weights_mat: NDArray,
    data_inputs: NDArray,
    data_outputs: NDArray,
    activation: str = "relu",
) -> float:
    return predict_outputs(weights_mat, data_inputs, data_outputs, activation)[0]
