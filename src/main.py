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


class StateIndex(enum.IntEnum):
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
        self._state_history[StateIndex.STATE, self._idx] = initial_state  # value
        self._state_history[StateIndex.TIME, self._idx] = 0.0  # time
        # self._state_history[2, self._current_index] #Reward
        self._harvest_history: list[HarvestOutcome] = []

    @property
    def state_history(self) -> np.ndarray:
        return self._state_history[:, : self._idx + 1]

    @property
    def harvest_history(self) -> list[HarvestOutcome]:
        return self._harvest_history

    def travel(self, step: float) -> Self:
        n_steps = int(step / self._dt)
        _drift = self._drift_rate * step + np.random.normal(
            0, self._noise_std * np.sqrt(n_steps)
        )
        self._state_history[StateIndex.STATE, self._idx + 1] = (
            self._state_history[StateIndex.STATE, self._idx] + _drift
        )
        self._state_history[StateIndex.TIME, self._idx + 1] = (
            self._state_history[StateIndex.TIME, self._idx] + n_steps * self._dt
        )
        self._idx += 1
        return self

    def harvest(self, patch: "Patch") -> Self:
        harvest = patch.harvest()
        self._state_history[StateIndex.STATE, self._idx + 1] = (
            self._state_history[StateIndex.STATE, self._idx] - harvest
        )
        self._state_history[StateIndex.TIME, self._idx + 1] = (
            self._state_history[StateIndex.TIME, self._idx] + self._dt
        )  # add a single dt
        self._state_history[StateIndex.REWARD, self._idx + 1] = harvest
        self._idx += 1
        self._harvest_history.append(
            HarvestOutcome(
                index=self._idx,
                value=self._state_history[StateIndex.STATE, self._idx],
                time=self._state_history[StateIndex.TIME, self._idx],
                reward=harvest,
                is_rejected=False,
            )
        )
        return self

    def has_ended(self):
        if (
            outcome := self._state_history[StateIndex.STATE, self._idx]
            >= self._threshold
        ):
            self.end()
        return outcome

    def end(self) -> Self:
        self._harvest_history.append(
            HarvestOutcome(
                index=self._idx,
                value=self._state_history[StateIndex.STATE, self._idx],
                time=self._state_history[StateIndex.TIME, self._idx],
                reward=np.nan,
                is_rejected=True,
            )
        )
        return self

    def plot(self):
        fig, axs = plt.subplots()
        axs.plot(
            self._state_history[StateIndex.TIME, : self._idx + 1],
            self._state_history[StateIndex.STATE, : self._idx + 1],
        )
        axs.scatter(
            self._state_history[StateIndex.TIME, : self._idx + 1],
            self._state_history[StateIndex.REWARD, : self._idx + 1],
        )
        return fig, axs


class Patch:
    def __init__(self, reward_amount: float = 0.1) -> None:
        self._reward_probability = [0.9 * (0.9**i) for i in range(20)]
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
        return (~np.isnan(self.ddm.state_history[StateIndex.REWARD, :])).sum()

    @property
    def n_sites_rewarded(self) -> int:
        return (self.ddm.state_history[StateIndex.REWARD, :] > 0).sum()

    @property
    def total_time(self) -> float:
        return self.ddm.state_history[StateIndex.TIME, -1]

    def consecutive_failures(self) -> np.ndarray:
        harvest_history = self.ddm.harvest_history
        reward_history = [harvest.reward > 0 for harvest in harvest_history]

        consecutive_fails = np.zeros(len(harvest_history), dtype=int)
        for i in range(1, len(reward_history)):
            c = 0
            for j in sorted(range(i), reverse=True):
                # transverse backwards until we hit a reward
                # the current outcome is not included
                if reward_history[j - 1]:
                    break
                c += 1
            consecutive_fails[i] = c
        return consecutive_fails

    def flattened_consecutive_failures_x_outcome_x_reward(self) -> np.ndarray:
        failures = self.consecutive_failures()
        outcomes = np.array([not h.is_rejected for h in self.ddm.harvest_history])
        # The cumulative reward is up to but not including the current harvest
        rewards = np.concatenate(
            [[0], np.cumsum([h.reward for h in self.ddm.harvest_history])[:-1]]
        )
        return np.stack([failures, outcomes, rewards], axis=1)


def main():
    N = 500000
    runs: list[RunResult] = []
    size_length_distribution = ExponentialDistribution(
        scale=1.5, min_bound=0.5, max_bound=4
    )
    for _ in track(range(N), description="Running simulations..."):
        run = RunResult(
            patch=Patch(reward_amount=0.03), ddm=DDM(drift_rate=0.25, noise_std=0.05)
        )
        while not run.ddm.has_ended():
            run.ddm.travel(size_length_distribution.sample()).harvest(run.patch)
        runs.append(run)

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(2, 3, 1)
    ax.hist([r.n_sites_visited for r in runs], bins=np.arange(0, 20, 1, dtype=int))
    ax.set_xlabel("Number of Sites Visited")
    ax.set_ylabel("Count")

    ax = fig.add_subplot(2, 3, 2)
    ax.hist([r.n_sites_rewarded for r in runs], bins=np.arange(0, 10, 1, dtype=int))
    ax.set_xlabel("Number of Sites Rewarded")
    ax.set_ylabel("Count")

    ax = fig.add_subplot(2, 3, 3)
    for r in runs[::100]:
        ax.plot(
            r.ddm.state_history[StateIndex.TIME, : r.ddm._idx + 1],
            r.ddm.state_history[StateIndex.STATE, : r.ddm._idx + 1],
            alpha=0.1,
            color="gray",
        )
        ax.axhline(r.ddm._threshold, color="red", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("DDM State (a.u.)")

    ax = fig.add_subplot(2, 3, 4)
    ax.hist([r.total_time for r in runs], bins=30)
    ax.set_xlabel("Total Time")
    ax.set_ylabel("Count")

    all_failures_x_outcome = [
        r.flattened_consecutive_failures_x_outcome_x_reward() for r in runs
    ]
    all_failures_x_outcome = np.concatenate(all_failures_x_outcome, axis=0)
    n_unique_failures = np.sort(np.unique(all_failures_x_outcome[:, 0]))
    n_unique_rewards = np.sort(np.unique(all_failures_x_outcome[:, 2]))
    n_unique_rewards = n_unique_rewards[~np.isnan(n_unique_rewards)]  # remove nan
    p_success = np.zeros((len(n_unique_failures), len(n_unique_rewards)))
    for row, n_fail in enumerate(n_unique_failures):
        for col, n_reward in enumerate(n_unique_rewards):
            trials = (all_failures_x_outcome[:, 0] == n_fail) & (
                all_failures_x_outcome[:, 2] == n_reward
            )

            outcomes = all_failures_x_outcome[trials, 1]
            p_success[row, col] = outcomes.sum() / len(outcomes)

    ax = fig.add_subplot(2, 3, 5)
    im = ax.imshow(
        p_success, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1
    )
    ax.set_ylabel("Consecutive Failures")
    ax.set_xlabel("Rewards collected")
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(2, 3, 6)
    ax.plot([0.9 * (0.9**i) for i in range(20)], marker="o")
    ax.set_xlabel("Reward  #")
    ax.set_ylabel("Reward Probability")
    ax.set_ylim(-0.05, 1.05)
    plt.savefig("foraging_simulation_results.png", dpi=300)


if __name__ == "__main__":
    main()
