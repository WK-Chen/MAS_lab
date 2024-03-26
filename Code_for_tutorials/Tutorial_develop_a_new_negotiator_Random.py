import random
from negmas.sao import SAONegotiator, SAOResponse
from negmas import Outcome, ResponseType

class MyRandomNegotiator(SAONegotiator):
    def __call__(self, state):
        offer = state.current_offer
        if offer is not None and self.ufun.is_not_worse(offer, None) and random.random() < 0.25 :
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        return SAOResponse(ResponseType.REJECT_OFFER, self.nmi.random_outcomes(1)[0])
