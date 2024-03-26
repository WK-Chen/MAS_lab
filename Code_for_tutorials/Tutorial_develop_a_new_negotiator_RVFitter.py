from scipy.optimize import curve_fit
from negmas.sao import SAONegotiator, SAOResponse
from negmas import Outcome, ResponseType
import numpy as np

def aspiration_function(t, mx, rv, e):
    """A monotonically decreasing curve starting at mx (t=0) and ending at rv (t=1)"""
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


class SimpleRVFitter(SAONegotiator):
    """A simple curve fitting modeling agent"""
    def __init__(self, *args, e: float = 5.0, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)
        self.e = e
        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of our last estimate of the opponent reserved value
        self._past_oppnent_rv = 0.0
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []

    def __call__(self, state):
        # update the opponent reserved value in self.opponent_ufun
        self.update_reserved_value(state.current_offer, state.relative_time)
        # run the acceptance strategy and if the offer received is acceptable, accept it
        if self.is_acceptable(state.current_offer, state.relative_time):
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        # call the offering strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.generate_offer(state.relative_time))

    def generate_offer(self, relative_time) -> Outcome:
        # The offering strategy
        # We only update our estimate of the rational list of outcomes if it is not set or
        # there is a change in estimated reserved value
        if (
            not self._rational
            or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
        ):
            # The rational set of outcomes sorted dependingly according to our utility function
            # and the opponent utility function (in that order).
            self._rational = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in self.nmi.outcome_space.enumerate_or_sample(
                        levels=10, max_cardinality=100_000
                    )
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > self.opponent_ufun.reserved_value
                ],
            )
        # If there are no rational outcomes (e.g. our estimate of the opponent rv is very wrong),
        # then just revert to offering our top offer
        if not self._rational:
            return self.ufun.best()
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        max_rational = len(self._rational) - 1
        indx = max(0, min(max_rational, int(asp * max_rational)))
        outcome = self._rational[indx][-1]
        return outcome

    def is_acceptable(self, offer, relative_time) -> bool:
        """The acceptance strategy"""
        # If there is no offer, there is nothing to accept
        if offer is None:
            return False
        # Find the current aspiration level
        asp = aspiration_function(
            relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        # accept if the utility of the received offer is higher than
        # the current aspiration
        return float(self.ufun(offer)) >= asp

    def update_reserved_value(self, offer, relative_time):
        """Learns the reserved value of the partner"""
        if offer is None:
            return
        # save to the list of utilities received from the opponent and their times
        self.opponent_utilities.append(float(self.opponent_ufun(offer)))
        self.opponent_times.append(relative_time)
        # Use curve fitting to estimate the opponent reserved value
        # We assume the following:
        # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
        # - The opponent never offers outcomes lower than their reserved value which means
        #   that their rv must be no higher than the worst outcome they offered for themselves.
        bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponent_utilities[0], rv, e
                ),
                self.opponent_times,
                self.opponent_utilities,
                bounds=bounds,
            )
            self._past_oppnent_rv = self.opponent_ufun.reserved_value
            self.opponent_ufun.reserved_value = optimal_vals[1]
        except Exception as e:
            pass