from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from bee_hive import BeeHive


class Bee:
    """Creates a bee object."""

    def __init__(self, bee_hive: "BeeHive") -> None:
        """Initialize a bee object with randomized solution vector.

        Args:
            bee_hive (BeeHive): Hive which this bee belongs to.
        """
        self.bee_hive: "BeeHive" = bee_hive

        # create a random solution vector
        self.vector = np.random.uniform(
            bee_hive.lower_bound, bee_hive.upper_bound, self.bee_hive.size
        )
        self.value: float = 0.0

        self.fit(
            input_data=self.bee_hive.input_train_data,
            output_data=self.bee_hive.output_train_data,
            replace_value=True,
        )

        # Initialize trial counter - i.e. abandonment counter
        self.trials: int = 0

    def fit(
        self,
        input_data: NDArray,
        output_data: NDArray,
        replace_value: bool = True,
    ) -> float:
        """Evaluate fitness of the solution vector.

        Args:
            input_data (NDArray): Testing or training input data.
            output_data (NDArray): Testing or training output data.
            replace_value (bool): Replace the value of the bee. Defaults to True.

        Returns:
            float: The value of the bee.
        """

        value: float = self.bee_hive.fit(
            self.vector_to_mat(), input_data, output_data, activation="sigmoid"
        )

        if replace_value:
            self.value = value

        return value

    def vector_to_mat(self) -> NDArray:
        """Convert 1D solution vector to an array of 2D matrices.

        Returns:
            NDArray: Array of 2D matrices representing the solution vector.
        """
        return np.array(
            [
                self.vector[: (self.bee_hive.dense_counts[index])].reshape(layer)
                for index, layer in enumerate(self.bee_hive.layers)
            ],
            dtype=object,
        )
