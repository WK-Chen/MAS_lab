import logging
from tqdm import tqdm, trange
import pickle
from train import inital_session, reformat_history

logging.basicConfig(level=logging.INFO)


def save_buffer_data(history, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(history, f)

def get_record(num_episodes):
    early = False
    history = []
    for episode in trange(num_episodes):
        session, pos = inital_session(early)
        session.run()

        state_history1, state_history2, action_history1, action_history2 = reformat_history(session, pos)
        print(state_history1)
        # assert False
        history.extend(state_history1)
        history.extend(state_history2)


    save_buffer_data(history, "buffer_record.pkl")


if __name__ == "__main__":
    num_episodes = 1000
    get_record(num_episodes)
