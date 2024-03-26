import itertools
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import matplotlib.pyplot as plt

import numpy as np
from negmas.helpers.misc import intin
from negmas.helpers.strings import unique_name
from negmas.inout import Scenario, UtilityFunction, pareto_frontier
from negmas.negotiators import Negotiator
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.preferences.generators import GENERATOR_MAP, generate_multi_issue_ufuns, generate_utility_values
from negmas.preferences.ops import nash_points
from negmas.preferences.value_fun import TableFun
from negmas.sao.mechanism import SAOMechanism
from negmas.tournaments.neg.simple import SimpleTournamentResults, cartesian_tournament

from anl.anl2024.negotiators.builtins import Boulware, Conceder, Linear, MiCRO, NashSeeker, RVFitter
from anl.anl2024.negotiators import Boulware, Conceder, RVFitter
# from anl.anl2024.negotiators.builtin import (
#     StochasticBoulware,
#     StochasticConceder,
#     StochasticLinear,
# NaiveTitForTat,
# )

__all__ = [
    "anl2024_tournament",
    "mixed_scenarios",
    "pie_scenarios",
    "arbitrary_pie_scenarios",
    "monotonic_pie_scenarios",
    "zerosum_pie_scenarios",
    "ScenarioGenerator",
    "DEFAULT_AN2024_COMPETITORS",
    "DEFAULT_TOURNAMENT_PATH",
    "DEFAULT2024SETTINGS",
]

DEFAULT_AN2024_COMPETITORS = (
    RVFitter,
    NashSeeker,
    MiCRO,
    Boulware,
    Conceder,
    Linear,
    # NaiveTitForTat,
    # StochasticLinear,
    # StochasticConceder,
    # StochasticBoulware
)
"""Default set of negotiators (agents) used as competitors"""

DEFAULT_TOURNAMENT_PATH = Path.home() / "negmas" / "anl2024" / "tournaments"
"""Default location to store tournament logs"""

ReservedRanges = tuple[tuple[float, ...], ...]


DEFAULT2024SETTINGS = dict(
    n_ufuns=2,
    n_scenarios=50,
    n_outcomes=(900, 1100),
    n_steps=(10, 10_000),
    n_repetitions=5,
    reserved_ranges=((0.0, 1.0), (0.0, 1.0)),
    competitors=DEFAULT_AN2024_COMPETITORS,
    rotate_ufuns=False,
    time_limit=60,
    pend=0,
    pend_per_second=0,
    step_time_limit=None,
    negotiator_time_limit=None,
    self_play=True,
    randomize_runs=True,
    known_partner=False,
    final_score=("advantage", "mean"),
    scenario_generator="mix",
    outcomes_log_uniform=True,
    generator_params=dict(
        reserved_ranges=((0.0, 1.0), (0.0, 1.0)),
        log_uniform=False,
        zerosum_fraction=0.05,
        monotonic_fraction=0.25,
        curve_fraction=0.25,
        pies_fraction=0.2,
        pareto_first=False,
        n_pareto=(0.005, 0.25),
    ),
)
"""Default settings for ANL 2024"""


ScenarioGenerator = Callable[[int, int | tuple[int, int] | list[int]], list[Scenario]]
"""Type of callable that can be used for generating scenarios. It must receive the number of scenarios and number of outcomes (as int, tuple or list) and return a list of `Scenario` s"""


def pies_scenarios(
    n_scenarios: int = 20,
    n_outcomes: int | tuple[int, int] | list[int] = 100,
    *,
    reserved_ranges: ReservedRanges = ((0.0, 0.999999), (0.0, 0.999999)),
    log_uniform: bool = True,
    monotonic=True,
) -> list[Scenario]:
    """Creates multi-issue scenarios with arbitrary/monotonically increasing value functions

    Args:
        n_scenarios: Number of scenarios to create
        n_outcomes: Number of outcomes per scenario. If a tuple it will be interpreted as a min/max range to sample n. outcomes from.
                    If a list, samples from this list will be used (with replacement).
        reserved_ranges: Ranges of reserved values for first and second negotiators
        log_uniform: If given, the distribution used will be uniform on the logarithm of n. outcomes (only used when n_outcomes is a 2-valued tuple).
        monotonic: If true all ufuns are monotonically increasing in the portion of the pie

    Remarks:
        - When n_outcomes is a tuple, the number of outcomes for each scenario will be sampled independently.
    """
    ufun_sets = []
    base_name = "DivideTyePies" if monotonic else "S"

    def normalize(x):
        mn, mx = x.min(), x.max()
        return ((x - mn) / (mx - mn)).tolist()

    def make_monotonic(x, i):
        x = np.sort(np.asarray(x), axis=None)

        if i:
            x = x[::-1]
        r = random.random()
        if r < 0.33:
            x = np.exp(x)
        elif r < 0.67:
            x = np.log(x)
        else:
            pass
        return normalize(x)

    max_jitter_level = 0.8
    for i in range(n_scenarios):
        n = intin(n_outcomes, log_uniform)
        issues = (
            make_issue(
                [f"{i}_{n-1 - i}" for i in range(n)],
                "portions" if not monotonic else "i1",
            ),
        )
        # funs = [
        #     dict(
        #         zip(
        #             issues[0].all,
        #             # adjust(np.asarray([random.random() for _ in range(n)])),
        #             generate(n, i),
        #         )
        #     )
        #     for i in range(2)
        # ]
        os = make_os(issues, name=f"{base_name}{i}")
        outcomes = list(os.enumerate_or_sample())
        ufuns = U.generate_bilateral(
            outcomes,
            conflict_level=0.5 + 0.5 * random.random(),
            conflict_delta=random.random(),
        )
        jitter_level = random.random() * max_jitter_level
        funs = [
            np.asarray([float(u(_)) for _ in outcomes])
            + np.random.random() * jitter_level
            for u in ufuns
        ]

        if monotonic:
            funs = [make_monotonic(x, i) for i, x in enumerate(funs)]
        else:
            funs = [normalize(x) for x in funs]
        ufuns = tuple(
            U(
                values=(TableFun(dict(zip(issues[0].all, vals))),),
                name=f"{uname}{i}",
                outcome_space=os,
                # reserved_value=(r[0] + random.random() * (r[1] - r[0] - 1e-8)),
            )
            for (uname, vals) in zip(("First", "Second"), funs)
            # for (uname, r, vals) in zip(("First", "Second"), reserved_ranges, funs)
        )
        sample_reserved_values(ufuns, reserved_ranges=reserved_ranges)
        ufun_sets.append(ufuns)

    return [
        Scenario(
            outcome_space=ufuns[0].outcome_space,  # type: ignore We are sure this is not None
            ufuns=ufuns,
        )
        for ufuns in ufun_sets
    ]


def pie_scenarios(
    n_scenarios: int = 20,
    n_outcomes: int | tuple[int, int] | list[int] = 100,
    *,
    reserved_ranges: ReservedRanges = ((0.0, 0.999999), (0.0, 0.999999)),
    log_uniform: bool = True,
    monotonic=False,
) -> list[Scenario]:
    """Creates single-issue scenarios with arbitrary/monotonically increasing utility functions

    Args:
        n_scenarios: Number of scenarios to create
        n_outcomes: Number of outcomes per scenario. If a tuple it will be interpreted as a min/max range to sample n. outcomes from.
                    If a list, samples from this list will be used (with replacement).
        reserved_ranges: Ranges of reserved values for first and second negotiators
        log_uniform: If given, the distribution used will be uniform on the logarithm of n. outcomes (only used when n_outcomes is a 2-valued tuple).
        monotonic: If true all ufuns are monotonically increasing in the portion of the pie

    Remarks:
        - When n_outcomes is a tuple, the number of outcomes for each scenario will be sampled independently.
    """
    ufun_sets = []
    base_name = "DivideTyePie" if monotonic else "S"

    def normalize(x):
        mn, mx = x.min(), x.max()
        return ((x - mn) / (mx - mn)).tolist()

    def make_monotonic(x, i):
        x = np.sort(np.asarray(x), axis=None)

        if i:
            x = x[::-1]
        r = random.random()
        if r < 0.33:
            x = np.exp(x)
        elif r < 0.67:
            x = np.log(x)
        else:
            pass
        return normalize(x)

    max_jitter_level = 0.8
    for i in range(n_scenarios):
        n = intin(n_outcomes, log_uniform)
        issues = (
            make_issue(
                [f"{i}_{n-1 - i}" for i in range(n)],
                "portions" if not monotonic else "i1",
            ),
        )
        # funs = [
        #     dict(
        #         zip(
        #             issues[0].all,
        #             # adjust(np.asarray([random.random() for _ in range(n)])),
        #             generate(n, i),
        #         )
        #     )
        #     for i in range(2)
        # ]
        os = make_os(issues, name=f"{base_name}{i}")
        outcomes = list(os.enumerate_or_sample())
        ufuns = U.generate_bilateral(
            outcomes,
            conflict_level=0.5 + 0.5 * random.random(),
            conflict_delta=random.random(),
        )
        jitter_level = random.random() * max_jitter_level
        funs = [
            np.asarray([float(u(_)) for _ in outcomes])
            + np.random.random() * jitter_level
            for u in ufuns
        ]

        if monotonic:
            funs = [make_monotonic(x, i) for i, x in enumerate(funs)]
        else:
            funs = [normalize(x) for x in funs]
        ufuns = tuple(
            U(
                values=(TableFun(dict(zip(issues[0].all, vals))),),
                name=f"{uname}{i}",
                outcome_space=os,
                # reserved_value=(r[0] + random.random() * (r[1] - r[0] - 1e-8)),
            )
            for (uname, vals) in zip(("First", "Second"), funs)
            # for (uname, r, vals) in zip(("First", "Second"), reserved_ranges, funs)
        )
        sample_reserved_values(ufuns, reserved_ranges=reserved_ranges)
        ufun_sets.append(ufuns)

    return [
        Scenario(
            outcome_space=ufuns[0].outcome_space,  # type: ignore We are sure this is not None
            ufuns=ufuns,
        )
        for ufuns in ufun_sets
    ]


def arbitrary_pie_scenarios(
    n_scenarios: int = 20,
    n_outcomes: int | tuple[int, int] | list[int] = 100,
    *,
    reserved_ranges: ReservedRanges = ((0.0, 0.999999), (0.0, 0.999999)),
    log_uniform: bool = True,
) -> list[Scenario]:
    return pie_scenarios(
        n_scenarios,
        n_outcomes,
        reserved_ranges=reserved_ranges,
        log_uniform=log_uniform,
        monotonic=False,
    )


def product(generator):
    """
    Calculates the product of all elements in a generator.

    Args:
        generator: A generator of numbers.

    Returns:
        The product of all elements in the generator.
    """
    total = 1
    for num in generator:
        total *= num
    return total


def find_three_integers(x):
    """Finds three integers that multiply to a number, considering numbers around x."""

    for offset in itertools.count(0):
        current_number = x + (-1) ** offset * offset

        # Stop searching downwards at 8
        if current_number < 8:
            break

        result = find_three_integers_for_number(current_number)
        if result:
            return result

    return None  # No solution found within the search range


def find_three_integers_for_number(x, fraction=0.1):
    """Helper function to find three integers for a given number."""

    # Check for perfect cubes
    cube_root = int(x ** (1 / 3))
    if cube_root**3 >= x * (1 - fraction):
        return (
            cube_root,
            cube_root + random.randint(-1, 1),
            cube_root,
        )

    # Factor x into primes
    prime_factors = []
    divisor = 2
    while x > 1:
        while x % divisor == 0:
            prime_factors.append(divisor)
            x //= divisor
        divisor += 1

    # Try to group prime factors into threes
    for combination in itertools.combinations(prime_factors, 3):
        if product(combination) == x:
            return combination

    # Try combining factors to create three integers
    if len(prime_factors) >= 4:
        for i in range(1, len(prime_factors) - 2):
            if (
                prime_factors[0] * prime_factors[i] * product(prime_factors[i + 1 :])
                == x
            ):
                return (
                    prime_factors[0],
                    prime_factors[i],
                    product(prime_factors[i + 1 :]),
                )

    return None


def monotonic_pies_scenarios(
    n_scenarios: int = 20,
    n_outcomes: int | tuple[int, int] | list[int] = 100,
    *,
    reserved_ranges: ReservedRanges = ((0.0, 0.999999), (0.0, 0.999999)),
    log_uniform: bool = False,
) -> list[Scenario]:
    ufun_sets = []
    for s in range(n_scenarios):
        ufuns = generate_multi_issue_ufuns(
            3,
            sizes=find_three_integers_for_number(n_outcomes),
            os_name=f"DivideThePies{s}",
        )
        os = ufuns[0].outcome_space
        assert os is not None
        sample_reserved_values(
            ufuns,
            pareto=tuple(tuple(u(_) for u in ufuns) for _ in os.enumerate_or_sample()),
            reserved_ranges=reserved_ranges,
        )
        ufun_sets.append(ufuns)

    return [
        Scenario(
            outcome_space=ufuns[0].outcome_space,  # type: ignore We are sure this is not None
            ufuns=ufuns,
        )
        for ufuns in ufun_sets
    ]


def monotonic_pie_scenarios(
    n_scenarios: int = 20,
    n_outcomes: int | tuple[int, int] | list[int] = 100,
    *,
    reserved_ranges: ReservedRanges = ((0.0, 0.999999), (0.0, 0.999999)),
    log_uniform: bool = True,
) -> list[Scenario]:
    return pie_scenarios(
        n_scenarios,
        n_outcomes,
        reserved_ranges=reserved_ranges,
        log_uniform=log_uniform,
        monotonic=True,
    )

    return [
        Scenario(
            outcome_space=ufuns[0].outcome_space,  # type: ignore We are sure this is not None
            ufuns=ufuns,
        )
        for ufuns in ufun_sets
    ]


def sample_reserved_values(
    ufuns: tuple[UtilityFunction, ...],
    pareto: tuple[tuple[float, ...], ...] | None = None,
    reserved_ranges: ReservedRanges = ((0.0, 1.0), (0.0, 1.0)),
    eps: float = 1e-3,
) -> tuple[float, ...]:
    """
    Samples reserved values that are guaranteed to allow some rational outcomes for the given ufuns and sets the reserved values.

    Args:
        ufuns: tuple of utility functions to sample reserved values for
        pareto: The pareto frontier. If not given, it will be calculated
        reserved_ranges: the range to sample reserved values from. Notice that the upper limit of this range will be updated
                         to ensure some rational outcoms
        eps: A small number indicating the absolute guaranteed margin of the sampled reserved value from the Nash point.

    """
    n_funs = len(ufuns)
    if pareto is None:
        pareto = pareto_frontier(ufuns)[0]
    assert pareto is not None, f"Cannot find the pareto frontier."
    nash = nash_points(ufuns, frontier=pareto, ranges=[(0, 1) for _ in range(n_funs)])
    if not nash:
        raise ValueError(
            f"Cannot find the Nash point so we cannot find the appropriate reserved ranges"
        )
    nash_utils = nash[0][0]
    if not reserved_ranges:
        reserved_ranges = tuple((0, 1) for _ in range(n_funs))
    reserved_ranges = tuple(
        tuple(min(r[_], n) for _ in range(n_funs))
        for n, r in zip(nash_utils, reserved_ranges)
    )
    reserved = tuple(
        r[0] + (r[1] - eps - r[0]) * random.random() for r in reserved_ranges
    )
    for u, r in zip(ufuns, reserved):
        u.reserved_value = float(r)
    return reserved


def zerosum_pie_scenarios(
    n_scenarios: int = 20,
    n_outcomes: int | tuple[int, int] | list[int] = 100,
    *,
    reserved_ranges: ReservedRanges = ((0.0, 0.499999), (0.0, 0.499999)),
    log_uniform: bool = True,
) -> list[Scenario]:
    """Creates scenarios all of the DivideThePie variety with proportions giving utility

    Args:
        n_scenarios: Number of scenarios to create
        n_outcomes: Number of outcomes per scenario (if a tuple it will be interpreted as a min/max range to sample n. outcomes from).
        reserved_ranges: Ranges of reserved values for first and second negotiators
        log_uniform: If given, the distribution used will be uniform on the logarithm of n. outcomes (only used when n_outcomes is a 2-valued tuple).

    Remarks:
        - When n_outcomes is a tuple, the number of outcomes for each outcome will be sampled independently
    """
    ufun_sets = []
    for i in range(n_scenarios):
        n = intin(n_outcomes, log_uniform)
        issues = (make_issue([f"{i}_{n-1 - i}" for i in range(n)], "portions"),)
        ufuns = tuple(
            U(
                values=(
                    TableFun(
                        {
                            _: float(int(str(_).split("_")[k]) / (n - 1))
                            for _ in issues[0].all
                        }
                    ),
                ),
                name=f"{uname}{i}",
                # reserved_value=(r[0] + random.random() * (r[1] - r[0] - 1e-8)),
                outcome_space=make_os(issues, name=f"DivideTyePie{i}"),
            )
            for k, uname in enumerate(("First", "Second"))
            # for k, (uname, r) in enumerate(zip(("First", "Second"), reserved_ranges))
        )
        sample_reserved_values(
            ufuns,
            pareto=tuple(
                tuple(u(_) for u in ufuns)
                for _ in make_os(issues).enumerate_or_sample()
            ),
            reserved_ranges=reserved_ranges,
        )
        ufun_sets.append(ufuns)

    return [
        Scenario(
            outcome_space=ufuns[0].outcome_space,  # type: ignore We are sure this is not None
            ufuns=ufuns,
        )
        for ufuns in ufun_sets
    ]


def mixed_scenarios(
    n_scenarios: int = DEFAULT2024SETTINGS["n_scenarios"],  # type: ignore
    n_outcomes: int
    | tuple[int, int]
    | list[int] = DEFAULT2024SETTINGS["n_outcomes"],  # type: ignore
    *,
    reserved_ranges: ReservedRanges = DEFAULT2024SETTINGS["reserved_ranges"],  # type: ignore
    log_uniform: bool = DEFAULT2024SETTINGS["outcomes_log_uniform"],  # type: ignore
    zerosum_fraction: float = DEFAULT2024SETTINGS["generator_params"]["zerosum_fraction"],  # type: ignore
    monotonic_fraction: float = DEFAULT2024SETTINGS["generator_params"]["monotonic_fraction"],  # type: ignore
    curve_fraction: float = DEFAULT2024SETTINGS["generator_params"]["curve_fraction"],  # type: ignore
    pies_fraction: float = DEFAULT2024SETTINGS["generator_params"]["pies_fraction"],  # type: ignore
    pareto_first: bool = DEFAULT2024SETTINGS["generator_params"]["pareto_first"],  # type: ignore
    n_ufuns: int = DEFAULT2024SETTINGS["n_ufuns"],  # type: ignore
    n_pareto: int | float | tuple[float | int, float | int] | list[int | float] = DEFAULT2024SETTINGS["generator_params"]["n_pareto"],  # type: ignore
    pareto_log_uniform: bool = False,
    n_trials=10,
) -> list[Scenario]:
    """Generates a mix of zero-sum, monotonic and general scenarios

    Args:
        n_scenarios: Number of scenarios to genearate
        n_outcomes: Number of outcomes (or a list of range thereof).
        reserved_ranges: the range allowed for reserved values for each ufun.
                         Note that the upper limit will be overridden to guarantee
                         the existence of at least one rational outcome
        log_uniform: Use log-uniform instead of uniform sampling if n_outcomes is a tuple giving a range.
        zerosum_fraction: Fraction of zero-sum scenarios. These are original DivideThePie scenarios
        monotonic_fraction: Fraction of scenarios where each ufun is a monotonic function of the received pie.
        curve_fraction: Fraction of general and monotonic scenarios that use a curve for Pareto generation instead of
                        a piecewise linear Pareto frontier.
        pies_fraction: Fraction of divide-the-pies multi-issue scenarios
        pareto_first: If given, the Pareto frontier will always be in the first set of outcomes
        n_ufuns: Number of ufuns to generate per scenario
        n_pareto: Number of outcomes on the Pareto frontier in general scenarios.
                Can be specified as a number, a tuple of a min and max to sample within, a list of possibilities.
                Each value can either be an integer > 1 or a fraction of the number of outcomes in the scenario.
        pareto_log_uniform: Use log-uniform instead of uniform sampling if n_pareto is a tuple
        n_trials: Number of times to retry generating each scenario if failures occures

    Returns:
        A list `Scenario` s
    """
    assert zerosum_fraction + monotonic_fraction <= 1.0
    nongeneral_fraction = zerosum_fraction + monotonic_fraction
    ufun_sets = []
    for i in range(n_scenarios):
        r = random.random()
        n = intin(n_outcomes, log_uniform)
        name = "S"
        if r < nongeneral_fraction:
            n_pareto_selected = n
            name = "DivideThePieGen"
        else:
            if isinstance(n_pareto, Iterable):
                n_pareto = type(n_pareto)(
                    int(_ * n + 0.5) if _ < 1 else int(_) for _ in n_pareto  # type: ignore
                )
            else:
                n_pareto = int(0.5 + n_pareto * n) if n_pareto < 1 else int(n_pareto)
            n_pareto_selected = intin(n_pareto, log_uniform=pareto_log_uniform)  # type: ignore
        ufuns, vals = None, None
        for _ in range(n_trials):
            try:
                if r < zerosum_fraction:
                    name = "DivideThePie"
                    vals = generate_utility_values(
                        n_pareto=n_pareto_selected,
                        n_outcomes=n,
                        n_ufuns=n_ufuns,
                        pareto_first=pareto_first,
                        pareto_generator="zero_sum",
                    )
                elif r < zerosum_fraction + pies_fraction:
                    name = "DivideThePies"
                    ufuns = generate_multi_issue_ufuns(
                        n_issues=3,
                        n_values=(9, 11),
                        pareto_generators=tuple(GENERATOR_MAP.keys()),
                        ufun_names=("First0", "Second1"),
                        os_name=f"{name}{i}",
                    )
                else:
                    if n_pareto_selected < 2:
                        n_pareto_selected = 2
                    vals = generate_utility_values(
                        n_pareto=n_pareto_selected,
                        n_outcomes=n,
                        n_ufuns=n_ufuns,
                        pareto_first=pareto_first,
                        pareto_generator="curve"
                        if random.random() < curve_fraction
                        else "piecewise_linear",
                    )
                break
            except:
                continue
        else:
            continue

        if ufuns is None:
            issues = (make_issue([f"{i}_{n-1 - i}" for i in range(n)], "portions"),)
            ufuns = tuple(
                U(
                    values=(
                        TableFun(
                            {_: float(vals[i][k]) for i, _ in enumerate(issues[0].all)}  # type: ignore
                        ),
                    ),
                    name=f"{uname}{i}",
                    # reserved_value=(r[0] + random.random() * (r[1] - r[0] - 1e-8)),
                    outcome_space=make_os(issues, name=f"{name}{i}"),
                )
                for k, uname in enumerate(("First", "Second"))
                # for k, (uname, r) in enumerate(zip(("First", "Second"), reserved_ranges))
            )
        sample_reserved_values(ufuns, reserved_ranges=reserved_ranges)
        ufun_sets.append(ufuns)

    return [
        Scenario(
            outcome_space=ufuns[0].outcome_space,  # type: ignore We are sure this is not None
            ufuns=ufuns,
        )
        for ufuns in ufun_sets
    ]


GENMAP = dict(
    monotonic=monotonic_pie_scenarios,
    arbitrary=arbitrary_pie_scenarios,
    zerosum=zerosum_pie_scenarios,
    pies=monotonic_pies_scenarios,
    default=mixed_scenarios,
    mix=mixed_scenarios,
)

DEFAULT2024GENERATOR = mixed_scenarios
"""Default generator type for ANL 2024"""


def generate_scenario(
    n_scenarios: int = DEFAULT2024SETTINGS["n_scenarios"],  # type: ignore
    n_outcomes: int | tuple[int, int] | list[int] = DEFAULT2024SETTINGS["n_outcomes"],  # type: ignore
    n_steps: int | tuple[int, int] | None = DEFAULT2024SETTINGS["n_steps"],  # type: ignore
    scenario_generator: str | ScenarioGenerator = DEFAULT2024SETTINGS["scenario_generator"],  # type: ignore
    generator_params=None,  # type: ignore

) -> SimpleTournamentResults:
    if generator_params is None:
        generator_params = DEFAULT2024SETTINGS["generator_params"]
    if isinstance(scenario_generator, str):
        scenario_generator = GENMAP[scenario_generator]

    scenarios = scenario_generator(n_scenarios, n_outcomes, **generator_params)

    private_infos = [
        tuple(
            dict(opponent_ufun=U(values=_.values, weights=_.weights, bias=_._bias, reserved_value=_.reserved_value, outcome_space=_.outcome_space))  # type: ignore
            for _ in s.ufuns[::-1]
        )
        for s in scenarios
    ]
    # print(private_infos)

    return scenarios[0], private_infos[0]

if __name__ == "__main__":
    scenario, private_info = generate_scenario(
        n_scenarios=1,
        n_outcomes=DEFAULT2024SETTINGS['n_outcomes'],
    )
    n_steps = random.randint(DEFAULT2024SETTINGS['n_steps'][0], DEFAULT2024SETTINGS['n_steps'][1])
    session = SAOMechanism(n_steps=n_steps, outcome_space=scenario.outcome_space)
    session.add(Boulware(name="seller", private_info=private_info[0]), ufun=scenario.ufuns[0])
    session.add(Boulware(name="seller", private_info=private_info[1]), ufun=scenario.ufuns[1])
    print(session.run())
    for i, _ in enumerate(session.history):
        print(f"{i}: {_.new_offers}")  # the first line gives the offer of the seller and the buyer  in the first round

    session.plot(ylimits=(0.0, 1.01), show_reserved=False, mark_max_welfare_points=False)
    plt.show()

    # # sample_pair = generate_sample()
    # # save_sample(sample_pair, "../data/Domain1")
    #
    # scenario = mixed_scenarios(n_scenarios=1, n_outcomes=3)[0]
    # session = SAOMechanism(n_steps=10, outcome_space=scenario.outcome_space)
    #
    # seller_utility = scenario.ufuns[0]
    # # print("seller utility")
    # # print(seller_utility.to_xml_str())
    # # print("=="*10)
    # buyer_utility = scenario.ufuns[1]
    # # print("buyer utility")
    # # print(buyer_utility.to_xml_str())
    # # print("==" * 10)
    # # create and add selller and buyer to the session
    # SACagent1 = SACAgent()
    # session.add(MyNegotiator(name="seller", SAC=SACagent1, private_info=dict(opponent_ufun=buyer_utility)),
    #             ufun=seller_utility)
    # SACagent2 = SACAgent()
    # session.add(MyNegotiator(name="buyer", SAC=SACagent2, private_info=dict(opponent_ufun=seller_utility)),
    #             ufun=buyer_utility)
    #
    # # run the negotiation and show the results
    # session.run()
    #
    # state_history1 = session.get_negotiator(session.negotiator_ids[0]).state_history
    # action_history1 = session.get_negotiator(session.negotiator_ids[0]).action_history
    # state_history2 = session.get_negotiator(session.negotiator_ids[1]).state_history
    # action_history2 = session.get_negotiator(session.negotiator_ids[1]).action_history
    #
    # if len(state_history1) > len(state_history2):
    #     state_history1.append(state_history1[-1])
    #     action_history1[-1] = 0.0
    #     state_history2.append(session.get_negotiator(session.negotiator_ids[1]).ufun(session.agreement))
    # else:
    #     state_history1.append(session.get_negotiator(session.negotiator_ids[0]).ufun(session.agreement))
    #     state_history2.append(state_history2[-1])
    #     action_history2[-1] = 0.0
    #
    # print(len(state_history1), len(action_history2), len(state_history2), len(action_history2))
    #
    # for i, _ in enumerate(session.history):
    #     print(f"{i}: {_.new_offers}")  # the first line gives the offer of the seller and the buyer  in the first round
    #
    # session.plot(ylimits=(0.0, 1.01), show_reserved=False, mark_max_welfare_points=False)
    # plt.show()