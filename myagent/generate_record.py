import logging
import random
from data_gen import generate_scenario, DEFAULT2024SETTINGS
from negmas.sao.mechanism import SAOMechanism
from anl.anl2024.negotiators import Boulware, Conceder, RVFitter, NashSeeker, MiCRO, Linear, NaiveTitForTat
from anl.anl2024.negotiators.builtins.wrappers import StochasticLinear, StochasticConceder, StochasticBoulware
from myagent import MyNegotiator
from tqdm import tqdm, trange
import pickle
from train import reformat_history

logging.basicConfig(level=logging.INFO)

TRAIN_COMPETITORS = (
    MyNegotiator,
    RVFitter,
    NashSeeker,
    MiCRO,
    Boulware,
    Conceder,
    Linear,
    NaiveTitForTat,
    StochasticLinear,
    StochasticConceder,
    StochasticBoulware,
)

def inital_session(opponent, model_path="model.pth"):
    scenario, private_info = generate_scenario(
        n_scenarios=1,
        n_outcomes=DEFAULT2024SETTINGS['n_outcomes'],
    )
    n_steps = random.randint(DEFAULT2024SETTINGS['n_steps'][0], 1000)

    session = SAOMechanism(n_steps=n_steps, outcome_space=scenario.outcome_space)

    if random.random() < 0.5:
        if opponent == MyNegotiator:
            session.add(
                MyNegotiator(name="seller", mode="trn", model_path=model_path, private_info=private_info[0]),
                ufun=scenario.ufuns[0])
        else:
            session.add(opponent(name="seller", private_info=private_info[0]), ufun=scenario.ufuns[0])
        session.add(MyNegotiator(name="buyer", mode="trn", model_path=model_path, private_info=private_info[1]),
                    ufun=scenario.ufuns[1])
        pos = [1]
    else:
        session.add(MyNegotiator(name="seller", mode="trn", model_path=model_path, private_info=private_info[0]),
                    ufun=scenario.ufuns[0])
        if opponent == MyNegotiator:
            session.add(
                MyNegotiator(name="seller", mode="trn", model_path=model_path, private_info=private_info[1]),
                ufun=scenario.ufuns[1])
        else:
            session.add(opponent(name="buyer", private_info=private_info[1]), ufun=scenario.ufuns[1])
        pos = [0]

    return session, pos

def save_data(history, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(history, f)

def generate_data(num_episodes):
    state_records = []
    new_opponent = random.choice(TRAIN_COMPETITORS)

    for episode in trange(num_episodes):
        new_opponent = random.choice(TRAIN_COMPETITORS)

        session, pos = inital_session(new_opponent)
        session.run()

        neg = session.get_negotiator(session.negotiator_ids[0]) if pos == [0] else (
            session.get_negotiator(session.negotiator_ids[1]))

        state_history, _, _ = reformat_history(session, neg)

        state_records.extend(state_history)

    save_data(state_records, "state_records.pkl")

if __name__ == '__main__':
    num_episodes = 100
    generate_data(num_episodes)