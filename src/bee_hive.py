import copy
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from sys import stdout
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .bee import Bee
from .config import Config
from .debug_tools import block_interruption


@dataclass
class ResultStats:
    iteration: int
    best: float
    mean: float
    best_test: float
    time: float
    values: list[float]


class BeeHive:
    """Creates an Artificial Bee Colony (ABC) algorithm.

    The population of the hive is composed of three distinct types of individuals:
        1. "employees",
        2. "onlookers",
        3. "scouts".

    The employed bees and onlooker bees exploit the nectar sources around the hive - i.e.
    exploitation phase - while the scouts explore the solution domain - i.e. exploration phase.

    The number of nectar sources around the hive is equal to the number of actively employed bees
    and the number of employeesis equal to the number of onlooker bees.
    """

    def __init__(
        self,
        shape: list[int],
        fitness_func: Callable[..., float],
        input_train_data: NDArray,
        output_train_data: NDArray,
        input_test_data: NDArray,
        output_test_data: NDArray,
    ) -> None:
        """Create a bee hive for the ABC algorithm.

        1. INITIALISATION PHASE.
        -----------------------
        The initial population of bees should cover the entire search space as much as possible by
        randomizing individuals within the search space constrained by the prescribed lower and
        upper bounds.

        Args:
            shape (list[int]): shape of the solution vector
            fitness_func (Callable[..., float]): fitness function
            input_train_data (NDArray): training input data for fitness function
            output_train_data (NDArray): training output data for fitness function
            input_test_data (NDArray): testing input data for fitness function
            output_test_data (NDArray): testing output data for fitness function
        """
        # assign properties of the optimisation problem
        self.fit: Callable[..., float] = fitness_func
        self.lower_bound: int = Config.min_value
        self.upper_bound: int = Config.max_value
        self.neurons: NDArray = np.array(shape)
        self.layers: list[tuple[int, int]] = np.stack(
            (self.neurons[:-1], self.neurons[1:]), axis=1
        ).tolist()
        self.dense_counts: list[int] = np.product(self.layers, axis=1)
        self.size: int = np.sum(self.dense_counts)
        self.mutation_percent: int = Config.mutation_probability

        # compute the number of employees
        self.numb_bees: int = Config.bees_count + (Config.bees_count % 2)

        # assign properties of algorithm
        self.max_itrs: int = Config.generations_count
        self.max_trials = Config.max_no_improvement_limit

        self.input_train_data: NDArray = input_train_data
        self.output_train_data: NDArray = output_train_data
        self.input_test_data: NDArray = input_test_data
        self.output_test_data: NDArray = output_test_data

        # create a bee hive
        self.population: NDArray = np.array([Bee(self) for _ in range(self.numb_bees)])

        # find currently best bee
        self.best: float = self._find_best_bee().value

        # compute selection probability
        self._compute_probability()

        self._prepare_logging()

    def _prepare_logging(self) -> None:
        """Prepare logging directory and open file sinks."""

        now = datetime.now()
        output_dir: str = f"{Config.output_path}/{now.date()}_{now.time().strftime('%H-%M-%S')}"

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.stats_file = open(f"{output_dir}/stats.txt", mode="w", encoding="utf-8")
        self.results_file = open(f"{output_dir}/results.txt", mode="w", encoding="utf-8")
        self._prepare_output_files()

    def _prepare_output_files(self) -> None:
        """Prepare headers of output data files."""

        data_header: str = (
            f"Bees: {self.numb_bees}; "
            f"mutation precent: {self.mutation_percent}; "
            f"max trials: {self.max_trials}\n"
        )

        self.results_file.write(data_header)
        self.stats_file.write(data_header)
        self.stats_file.write("iter best mean test time\n")

    def run(self) -> None:
        """Run an Artificial Bee Colony (ABC) algorithm."""

        with block_interruption():
            for itr in range(self.max_itrs):
                start_time: float = time.time()

                # employees phase
                for index in range(self.numb_bees):
                    self._send_employee(index)

                # onlookers phase
                self._send_onlookers()

                # scouts phase
                self._send_scout()

                # compute best path
                best_bee = self._find_best_bee()
                self.best = max(self.best, best_bee.value)

                end_time: float = time.time()

                # cumulate statistics of this iteration
                values: list[float] = [bee.value for bee in self.population]
                stats = ResultStats(
                    iteration=itr,
                    best=self.best,
                    mean=sum(values) / self.numb_bees,
                    best_test=best_bee.fit(self.input_test_data, self.output_test_data, False),
                    time=end_time - start_time,
                    values=values,
                )
                self._save_to_file(stats)

                # print out information about current iteration to stdout
                if Config.log_iterations:
                    self._log(stats)

        self.stats_file.close()
        self.results_file.close()

    def _find_best_bee(self) -> Bee:
        """Finds current best bee candidate."""

        # retrieve fitness of bees within the hive
        values: list[float] = [bee.value for bee in self.population]
        index: int = values.index(max(values))

        # return bee with the best fitness
        return self.population[index]

    def _compute_probability(self) -> NDArray:
        """
        Computes the relative chance that a given solution vector is chosen by an onlooker bee after
        the Waggle dance ceremony when employed bees are back within the hive.
        """

        # retrieve fitness of bees within the hive
        values: NDArray = np.array([bee.value for bee in self.population])

        # compute probabilities the way Karaboga does in his classic ABC implementation
        self.probabilities = values * 0.9 / np.max(values) + 0.1

        # return cumulative sums of probabilities
        self.cumulative_probabilities = np.cumsum(self.probabilities)

    def _send_employee(self, bee_index: int, other_bee_index: int | None = None) -> None:
        """2. SEND EMPLOYED BEES PHASE.

        During this 2nd phase, new candidate solutions are produced for each employed bee by
        mutation and mutation of the employees.

        If the modified vector of the mutant bee solution is better than that of the original bee,
        the new vector is assigned to the bee.
        """

        # save a copy of the bee in case it won't improve after mutation
        bee: Bee = self.population[bee_index]
        parent_bee_vector: NDArray = bee.vector.copy()
        parent_bee_value: float = copy.copy(bee.value)

        # select another bee
        if other_bee_index is None:
            other_bee_index = (rand := np.random.choice(self.numb_bees - 1)) + (rand >= bee_index)

        # mutate a bee from itself and any other bee
        for i in np.random.choice(self.size, int(self.size * (self.mutation_percent / 100))):
            bee.vector[i] = self._mutate(i, bee_index, other_bee_index)

        # recompute fitness of the mutated bee
        bee.fit(
            input_data=self.input_train_data,
            output_data=self.output_train_data,
            replace_value=True,
        )

        # deterministic crowding: keep the results if mutated bee is better than the original
        if bee.value > parent_bee_value:
            bee.trials = 0
        else:
            bee.vector = parent_bee_vector
            bee.value = parent_bee_value
            bee.trials += 1

    def _send_onlookers(self) -> None:
        """3. SEND ONLOOKERS PHASE.

        We define as many onlooker bees as there are employed bees in the hive. Onlooker bees
        will attempt to locally improve the solution path of the employed bee which is selected by roulette wheel.
        """

        max_probability: float = self.cumulative_probabilities[-1]
        # draw a random number from U[0, max_probability]
        phi = random.random() * max_probability

        better_index = (self.cumulative_probabilities >= phi).nonzero()[0][0]

        # send onlookers
        for index in range(self.numb_bees):
            # send new onlooker
            self._send_employee(index, better_index)

    def _send_scout(self) -> None:
        """4. SEND SCOUT BEE PHASE.

        Identifies bees whose abandonment counts exceed preset trials limit, abandons it and creates
        a new random bee to explore new random area of the domain space.

        In real life, after the depletion of a food nectar source, a bee moves on to other food
        sources. By this means, the employed bee which cannot improve their solution until the
        abandonment counter reaches the limit of trials becomes a scout bee.

        Therefore, scout bees in ABC algorithm prevent stagnation of employed bee population.

        Intuitively, this method provides an easy means to overcome any local optima within which
        a bee may have been trapped.
        """

        # retrieve the number of trials for all bees
        trials = np.array([bee.trials for bee in self.population])

        # identify bees with exceeding number of trials
        indexes = (trials > self.max_trials).nonzero()[0]

        # replace bees that reached the limit of trials
        for index in indexes:
            # create a new scout bee randomly
            self.population[index] = Bee(self)

            # send scout bee to exploit its solution vector
            self._send_employee(index)

    def _mutate(self, dim: int, bee_index: int, other_bee_index: int) -> float:
        """Mutate a given solution vector.

        Args:
            dim (int): vector's dimension to be mutated
            bee_index (int): index of the current bee that will be mutated
            other_bee_index (int): index of another bee

        Returns:
            float: mutated weight passed to solution vector.
        """
        bee: Bee = self.population[bee_index]
        other_bee: Bee = self.population[other_bee_index]

        factor: float = (random.random() - 0.5) * 2
        mutated_value: float = bee.vector[dim] + factor * (bee.vector[dim] - other_bee.vector[dim])

        mutated_value = max(mutated_value, self.lower_bound)
        mutated_value = min(mutated_value, self.upper_bound)

        return mutated_value

    def _save_to_file(self, stats: ResultStats) -> None:
        """Save iteration results to files.

        Args:
            stats (ResultStats): Cumulated statistics of the current iteration.
        """
        self.stats_file.write(
            f"{stats.iteration} {stats.best} {stats.mean} {stats.best_test} {stats.time}\n"
        )
        self.results_file.write(f"{stats.values}\n")

    def _log(self, stats: ResultStats) -> None:
        """Print out current iteration results to stdout.

        Args:
            stats (ResultStats): Cumulated statistics of the current iteration.
        """

        stdout.write(
            f"[{stats.iteration}] "
            f"Best Eval = {stats.best:.4f} | "
            f"Mean Eval = {stats.mean:.4f} | "
            f"Test Eval = {stats.best_test:.4f} | "
            f"Time = {stats.time:.2f}s\n"
        )
