"""
This is the Code that is part of Tutorial 1 for the ANL 2024 competition, see URL.

This Code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

from negmas import (
    make_issue,
    SAOMechanism
)
from anl.anl2024.negotiators import Boulware, Conceder, RVFitter
from negmas.preferences import LinearAdditiveUtilityFunction as UFun
from negmas.preferences.value_fun import IdentityFun, AffineFun
import matplotlib.pyplot as plt
# from myagent.myagent import MyNegotiator

def main():
    # create negotiation agenda (issues)
    issues = [
        make_issue(name="price", values=50),

    ]

    # create the mechanism
    session = SAOMechanism(issues=issues, n_steps=20)

    seller_utility = UFun(
        values=[IdentityFun()],
        outcome_space=session.outcome_space,
    )

    buyer_utility = UFun(
        values=[AffineFun(slope=-1)],
        outcome_space=session.outcome_space,
    )

    seller_utility = seller_utility.normalize()
    buyer_utility = buyer_utility.normalize()

    # create and add selller and buyer to the session
    session.add(Boulware(name="seller"), ufun=seller_utility)
    session.add(Boulware(name="buyer"), ufun=buyer_utility)

    # run the negotiation and show the results
    # print(session.run())
    for i in range(100):
        print(session.step().running)
    # negotiation history
    for i, _ in enumerate(session.history):
        print(f"{i:03}: {_.new_offers}")  # the first line gives the offer of the seller and the buyer  in the first round

    session.plot(ylimits=(0.0, 1.01), show_reserved=False, mark_max_welfare_points=False)
    plt.show()

if __name__ == "__main__":
    main()


