"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This Code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""
import logging
import random

import negmas
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
from sac import SACContinuous
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib
from pathlib import Path
import torch
import numpy as np
from anl.anl2024 import anl2024_tournament
from anl.anl2024.negotiators import Boulware, Conceder, RVFitter
import matplotlib.pyplot as plt
from functools import partial

logging.basicConfig(level=logging.INFO)

class MyNegotiator(SAONegotiator):
    def __init__(self, *args, **kwargs):
        self.mode = kwargs.pop("mode")
        self.model_path = kwargs.pop("model_path")
        if isinstance(self.model_path, str):
            if not Path(self.model_path).exists():
                self.SACagent = SACContinuous()
            else:
                self.SACagent = torch.load(self.model_path)
        elif isinstance(self.model_path, SACContinuous):
            self.SACagent = self.model_path
        else:
            assert False

        super().__init__(*args, **kwargs)

        if self.mode == 'test':
            self.rv_predict = None
            # self.rv_predict = joblib.load('regressor_model.pkl')
        else:
            self.rv_predict = None

        self._past_oppnent_rv = 0.0
        self.history_utility_op_offer_op = []
        self.history_utility_self_offer_op = []
        self.history_utility_self_offer_self = []
        self.state_history = []
        self.action_history = []
        self.partner_reserved_value = 0.0

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        # self.nash_point = self.nash_points()
        self.history_utility_op_offer_op.extend([self.opponent_ufun.max()]*2)
        self.history_utility_self_offer_self.extend([self.ufun.max()]*3)
        # print(f"Agent: {self.nash_point}")
        # print(self.ufun.max())
        # print(f"MinMAX.ufun: {self.ufun.minmax(self.ufun.outcome_space)}")
        # assert False
        # print(f"MinMAX.op_ufun: {self.opponent_ufun.minmax(self.ufun.outcome_space)}")

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) >= self.ufun.reserved_value
        ]

        # Estimate the reservation value, as a first guess, the opponent has the same reserved_value as you
        self.partner_reserved_value = self.ufun.reserved_value

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Called to (counter-)offer.

        Args:
            state: the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc).
        Returns:
            A response of type `SAOResponse` which indicates whether you accept, or reject the offer or leave the negotiation.
            If you reject an offer, you are required to pass a counter offer.

        Remarks:
            - This is the ONLY function you need to implement.
            - You can access your ufun using `self.ufun`.
            - You can access the opponent's ufun using self.opponent_ufun(offer)
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
              - If this is `None`, you are starting the negotiation now (no offers yet).
        """
        self.op_bid = state.current_offer

        if self.op_bid is None:
            self.history_utility_op_offer_op.append(self.opponent_ufun.max())
            # self.history_utility_self_offer_op.append(self.ufun(self.opponent_ufun.extreme_outcomes()[1]))
        else:
            self.history_utility_op_offer_op.append(self.opponent_ufun(self.op_bid))
            # self.history_utility_self_offer_op.append(self.ufun(self.op_bid))

        self.update_partner_reserved_value(state)


        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        self.my_bid = self.bidding_strategy(state)
        # self.history_utility_op_offer_self.append(self.opponent_ufun(self.my_bid))
        self.history_utility_self_offer_self.append(self.ufun(self.my_bid))

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, self.op_bid)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.my_bid)

    def nash_points(self):
        frontier, frontier_outcomes = negmas.pareto_frontier(ufuns=(self.ufun, self.opponent_ufun))
        point = negmas.nash_points(ufuns=(self.ufun, self.opponent_ufun), frontier=frontier)
        if not isinstance(point[0], tuple):
            print(point)
        point = point[0][0]
        return point

    def acceptance_strategy(self, state: SAOState) -> bool:
        assert self.ufun

        # if state.relative_time < 0.4 and self.ufun(self.op_bid) < self.nash_points()[0]:
        #     return False

        if self.ufun(self.op_bid) >= self.ufun(self.my_bid):
            return True

        # if state.relative_time > 0.8 and self.ufun(self.op_bid) > self.nash_points()[0] - 1e-4:
        #     return True

        # if state.relative_time > 0.95 and self.ufun(self.op_bid) > self.reserved_value:
        #     return True

        return False

    def get_state_space(self, state):
        if self.mode == 'test':
            return (
                0 if state.relative_time <= 0.75 else 1,
                # self.nash_point[0],
                # self.nash_point[1],
                self.history_utility_op_offer_op[-3],
                self.history_utility_op_offer_op[-2],
                self.history_utility_op_offer_op[-1],
                # self.history_utility_self_offer_op[-1],
                self.history_utility_self_offer_self[-3],
                self.history_utility_self_offer_self[-2],
                self.history_utility_self_offer_self[-1],
                self.reserved_value,
                # self.partner_reserved_value,
                self.opponent_ufun.reserved_value
            )
        else:
            return (
                0 if state.relative_time <= 0.75 else 1,
                # self.nash_point[0],
                # self.nash_point[1],
                self.history_utility_op_offer_op[-3],
                self.history_utility_op_offer_op[-2],
                self.history_utility_op_offer_op[-1],
                # self.history_utility_self_offer_op[-1],
                self.history_utility_self_offer_self[-3],
                self.history_utility_self_offer_self[-2],
                self.history_utility_self_offer_self[-1],
                self.reserved_value,
                self.opponent_ufun.reserved_value
            )

    def find_closest_value(self, dict, value):
        closest_key = None
        min_diff = float('inf')

        for key in dict.keys():
            diff = (key - value) ** 2
            if diff < min_diff:
                min_diff = diff
                closest_key = key

        return closest_key

    def action_to_bid(self, action):
        bid_value_dict = {_: self.ufun(_) for _ in self.rational_outcomes}
        value_bid_dict = {}
        for k, v in bid_value_dict.items():
            if v in value_bid_dict:
                value_bid_dict[v].append(k)
            else:
                value_bid_dict[v] = [k]
        next_value = self.history_utility_self_offer_self[-1] + action
        closest_value = self.find_closest_value(value_bid_dict, next_value)
        return value_bid_dict[closest_value]

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        action = self.SACagent.take_action(self.get_state_space(state))

        self.state_history.append(self.get_state_space(state))
        self.action_history.append(action)
        my_bids = self.action_to_bid(action[0])
        return random.choice(my_bids)

    def update_partner_reserved_value(self, state: SAOState) -> None:
        assert self.ufun and self.opponent_ufun
        if self.mode == "trn":
            return 0.0
        else:
            return 0.0
        info = (
            state.relative_time,
            self.nash_point[0],
            self.nash_point[1],
            self.history_utility_op_offer_op[-1],
            self.history_utility_self_offer_op[-1],
            self.history_utility_self_offer_self[-1],
            self.reserved_value
        )
        self.partner_reserved_value = self.rv_predict.predict(np.array(info).reshape(1, -1))[0]


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament
    run_a_tournament(partial(MyNegotiator, mode="test", model_path="model.pth"),
                     n_repetitions=1,
                     n_outcomes=1000,
                     n_scenarios=3,
                     small=True, debug=True,nologs=True)
    # results = anl2024_tournament(
    #     n_scenarios=1, n_repetitions=1, nologs=True, njobs=-1, verbosity=2,
    #     competitors=[MyNegotiator, Boulware]
    # )
    #
    # fig, ax = plt.subplots(figsize=(8, 6))
    # df = results.scores
    # for label, data in df.groupby('strategy'):
    #     data.advantage.plot(kind="kde", ax=ax, label=label)
    # plt.ylabel("advantage")
    # plt.legend()
    # plt.show()
    #
    # fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    # for i, col in enumerate(["advantage", "welfare", "nash_optimality"]):
    #     results.scores.groupby("strategy")[col].mean().sort_index().plot(kind="bar", ax=axs[i])
    #     axs[i].set_ylabel(col)
    #
    # plt.show()