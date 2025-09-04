import numpy as np
from typing import Self
from matplotlib import pyplot as plt
import enum
import dataclasses
from rich.progress import track


class ExponentialDistribution:
    def __init__(
        self, scale: float = 1.0, min_bound: float = 0.5, max_bound: float = 5
    ) -> None:
        def _sample() -> float:
            drawn = np.random.exponential(scale=scale)
            return max(min_bound, min(drawn, max_bound))

        self._distribution = _sample

    def sample(self) -> float:
        return self._distribution()


class Indicies(enum.IntEnum):
    STATE = 0
    TIME = 1
    REWARD = 2


@dataclasses.dataclass
class HarvestOutcome:
    index: int
    value: float
    time: float
    reward: float
    is_rejected: bool

class DDM:
    def __init__(
        self,
        initial_state: float = 0,
        drift_rate: float = 0.1,
        threshold: float = 1.0,
        noise_std: float = 0.005,
        dt: float = 0.1,
        _buffer_max_size: int = 10000,
    ) -> None:
        self._drift_rate = drift_rate
        self._threshold = threshold
        self._noise_std = noise_std
        self._dt = dt
        self._state_history: np.ndarray = np.full((3, _buffer_max_size), np.nan)
        self._idx = 0
        self._state_history[Indicies.STATE, self._idx] = initial_state  # value
        self._state_history[Indicies.TIME, self._idx] = 0.0  # time
        # self._state_history[2, self._current_index] #Reward
        self._choice_history: list[HarvestOutcome] = []

    @property
    def state_history(self) -> np.ndarray:
        return self._state_history[:, : self._idx + 1]

    def drift(self, step: float) -> Self:
        n_steps = int(step / self._dt)
        _drift = self._drift_rate * step + np.random.normal(
            0, self._noise_std * np.sqrt(n_steps)
        )
        self._state_history[Indicies.STATE, self._idx + 1] = (
            self._state_history[Indicies.STATE, self._idx] + _drift
        )
        self._state_history[Indicies.TIME, self._idx + 1] = (
            self._state_history[Indicies.TIME, self._idx] + n_steps * self._dt
        )
        self._idx += 1
        return self

    def harvest(self, patch: "Patch") -> Self:
        harvest = patch.harvest()
        self._state_history[Indicies.STATE, self._idx + 1] = (
            self._state_history[Indicies.STATE, self._idx] - harvest
        )
        self._state_history[Indicies.TIME, self._idx + 1] = (
            self._state_history[Indicies.TIME, self._idx] + self._dt
        )  # add a single dt
        self._state_history[Indicies.REWARD, self._idx + 1] = harvest
        self._idx += 1
        self._choice_history.append(HarvestOutcome(
            index=self._idx,
            value=self._state_history[Indicies.STATE, self._idx],
            time=self._state_history[Indicies.TIME, self._idx],
            reward=harvest,
            is_rejected=False,
        ))
        return self

    def has_ended(self):
        self.end()
        return self._state_history[Indicies.STATE, self._idx] >= self._threshold

    def end(self) -> Self:
        self._choice_history.append(HarvestOutcome(
            index=self._idx,
            value=self._state_history[Indicies.STATE, self._idx],
            time=self._state_history[Indicies.TIME, self._idx],
            reward=np.nan,
            is_rejected=True,
        ))
        return self

    def plot(self):
        fig, axs = plt.subplots()
        axs.plot(
            self._state_history[Indicies.TIME, : self._idx + 1],
            self._state_history[Indicies.STATE, : self._idx + 1],
        )
        axs.scatter(
            self._state_history[Indicies.TIME, : self._idx + 1],
            self._state_history[Indicies.REWARD, : self._idx + 1],
        )
        return fig, axs


class Patch:
    def __init__(self, reward_amount: float = 0.1) -> None:
        self._reward_probability = [0.9, 0.7, 0.6, 0.5, 0.42, 0.3]
        self._reward_amount = reward_amount

    def harvest(self) -> float:
        if self._reward_probability:
            if np.random.rand() < self._reward_probability[0]:
                self._reward_probability.pop(0)
                return self._reward_amount
        return 0.0


@dataclasses.dataclass
class RunResult:
    patch: Patch
    ddm: DDM

    @property
    def n_sites_visited(self) -> int:
        return (~np.isnan(self.ddm.state_history[Indicies.REWARD, :])).sum()

    @property
    def n_sites_rewarded(self) -> int:
        return (self.ddm.state_history[Indicies.REWARD, :] > 0).sum()




def main():
    N = 10000
    runs: list[RunResult] = []
    size_length_distribution = ExponentialDistribution(
        scale=1.0, min_bound=0.5, max_bound=5
    )
    for i in track(range(N), description="Running simulations..."):
        run = RunResult(
            patch=Patch(reward_amount=0.01), ddm=DDM(drift_rate=0.25, noise_std=0.05)
        )
        while not run.ddm.has_ended():
            size_length = size_length_distribution.sample()
            run.ddm.drift(size_length).harvest(run.patch)
        runs.append(run)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].hist([r.n_sites_visited for r in runs], bins=np.arange(0, 20, 1, dtype=int))
    axs[0].set_xlabel("Number of Sites Visited")
    axs[0].set_ylabel("Count")

    axs[1].hist([r.n_sites_rewarded for r in runs], bins=np.arange(0, 10, 1, dtype=int))
    axs[1].set_xlabel("Number of Sites Rewarded")
    axs[1].set_ylabel("Count")

    for r in runs[::100]:
        axs[2].plot(
            r.ddm.state_history[Indicies.TIME, : r.ddm._idx + 1],
            r.ddm.state_history[Indicies.STATE, : r.ddm._idx + 1],
            alpha=0.1,
            color="gray",
        )
        axs[2].axhline(r.ddm._threshold, color="red", linestyle="--")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("DDM State (a.u.)")
    plt.show()


if __name__ == "__main__":
    main()
