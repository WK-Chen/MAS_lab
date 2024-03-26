
from negmas import SAOMechanism
from anl.anl2024.negotiators import Conceder, Boulware
from anl.anl2024 import zerosum_pie_scenarios, monotonic_pie_scenarios, arbitrary_pie_scenarios
import matplotlib.pyplot as plt

def main():
    scenario = monotonic_pie_scenarios(n_scenarios=2, n_outcomes=20)[0] #zerosum_pie_scenarios, arbitrary_pie_scenarios
    session = SAOMechanism(issues=scenario.issues, n_steps=30)
    A_utility = scenario.ufuns[0]
    B_utility = scenario.ufuns[1]
    visualize((session, A_utility, B_utility))


def visualize(negotiation_setup):
    (session, A_utility, B_utility) = negotiation_setup

    # create and add selller and buyer to the session
    AgentA = Boulware(name="A")
    AgentB = Boulware(name="B")
    session.add(AgentA, ufun=A_utility)
    session.add(AgentB, ufun=B_utility)

    # run the negotiation and show the results. If omitted, just the scenario is shown.
    session.run()

    session.plot(ylimits=(0.0, 1.01), show_reserved=True)
    plt.show()

if __name__ == "__main__":
    main()