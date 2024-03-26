from anl.anl2024 import anl2024_tournament
from anl.anl2024.negotiators import Boulware, Conceder, RVFitter
from Code_for_tutorials.Tutorial_develop_a_new_negotiator_Random import MyRandomNegotiator
from Code_for_tutorials.Tutorial_develop_a_new_negotiator_RVFitter import SimpleRVFitter
import copy
from negmas.sao import SAOMechanism
from anl.anl2024.runner import mixed_scenarios
from anl.anl2024.negotiators.builtins import Linear
from matplotlib import pyplot as plt

#testing the agent
def run_tournament():
    results = anl2024_tournament(
        n_scenarios=1, n_repetitions=3, nologs=True, njobs=-1,
        competitors=[MyRandomNegotiator, Boulware]
        #competitors =[MyRandomNegotiator, SimpleRVFitter, Boulware, Conceder]
    )
    #visualization
    results.final_scores

    #plot to compare strategies:
    fig, ax = plt.subplots(figsize=(8, 6))
    df = results.scores
    for label, data in df.groupby('strategy'):
        data.advantage.plot(kind="kde", ax=ax, label=label)
    plt.ylabel("advantage")
    plt.legend();

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for i, col in enumerate(["advantage", "welfare", "nash_optimality"]):
        results.scores.groupby("strategy")[col].mean().sort_index().plot(kind="bar", ax=axs[i])
        axs[i].set_ylabel(col)

def run_negotiation():
    # create a scenario
    s = mixed_scenarios(1)[0]
    # copy ufuns and set rv to 0 in the copies
    ufuns0 = [copy.deepcopy(u) for u in s.ufuns]
    for u in ufuns0:
        u.reserved_value = 0.0
    # create the negotiation mechanism
    session = SAOMechanism(n_steps=1000, outcome_space=s.outcome_space)
    # add negotiators. Remember to pass the opponent_ufun in private_info
    session.add(
        SimpleRVFitter(name="SimpleRVFitter",
                       private_info=dict(opponent_ufun=ufuns0[1]))
        , ufun=s.ufuns[0]
    )
    session.add(Linear(name="Linear"), ufun=s.ufuns[1])
    #session.add(
    #RVFitter(name="RVFitter",
     #              private_info=dict(opponent_ufun=ufuns0[0]))
    #, ufun=s.ufuns[1]
    #)

    # run the negotiation and plot the results
    session.run()
    session.plot()
    plt.show()

if __name__ == "__main__":

    run_tournament()
    #run_negotiation()
