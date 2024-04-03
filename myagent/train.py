import logging
import os
import random
from sac import SACContinuous, ReplayBuffer
from data_gen import generate_scenario, DEFAULT2024SETTINGS
from negmas.sao.mechanism import SAOMechanism
from anl.anl2024.negotiators import Boulware, Conceder, RVFitter, NashSeeker, MiCRO, Linear, NaiveTitForTat
from myagent import MyNegotiator
from tqdm import tqdm, trange
import torch
from pathlib import Path
from helpers.runner import run_a_tournament
import matplotlib.pyplot as plt
import pickle
import math
from functools import partial
import shutil

logging.basicConfig(level=logging.INFO)


# TRAIN_COMPETITORS = all_negotiator_types()
TRAIN_COMPETITORS = (
    MyNegotiator,
    # Boulware,
    # Conceder,
    # RVFitter,
    # NashSeeker,
    # MiCRO,
    # Linear,
)

def inital_session(opponent, SACagent, path="model.pth"):
    scenario, private_info = generate_scenario(
        n_scenarios=1,
        n_outcomes=DEFAULT2024SETTINGS['n_outcomes'],
    )
    n_steps = random.randint(DEFAULT2024SETTINGS['n_steps'][0], 1000)
    # n_steps = random.randint(DEFAULT2024SETTINGS['n_steps'][0], DEFAULT2024SETTINGS['n_steps'][1])

    session = SAOMechanism(n_steps=n_steps, outcome_space=scenario.outcome_space)

    if random.random() < 0.5:
        if opponent == MyNegotiator:
            session.add(
                MyNegotiator(name="seller", mode="trn", model_path=path, private_info=private_info[0]),
                ufun=scenario.ufuns[0])
        else:
            session.add(opponent(name="seller", private_info=private_info[0]), ufun=scenario.ufuns[0])
        session.add(MyNegotiator(name="buyer", mode="trn", model_path=SACagent, private_info=private_info[1]),
                    ufun=scenario.ufuns[1])
        pos = [1]
    else:
        session.add(MyNegotiator(name="seller", mode="trn", model_path=SACagent, private_info=private_info[0]),
                    ufun=scenario.ufuns[0])
        if opponent == MyNegotiator:
            session.add(
                MyNegotiator(name="buyer", mode="trn", model_path=path, private_info=private_info[1]),
                ufun=scenario.ufuns[1])
        else:
            session.add(opponent(name="buyer", private_info=private_info[1]), ufun=scenario.ufuns[1])
        pos = [0]

    return session, pos


def reformat_history(session, neg):
    state_history, action_history = [], []
    his_o_o = neg.history_utility_op_offer_op
    his_s_s = neg.history_utility_self_offer_self
    his_o_o.append(neg.opponent_ufun(session.agreement))
    if session.state.step % 2 == 0:
        his_s_s[-1] = neg.ufun(session.agreement)
    assert len(his_o_o) == len(his_s_s)

    his = neg.state_history
    if len(his) == 0:
        return None, None, True
    time_step = [state[0] for state in his] + [his[-1][0]]
    rvs = his[0][-2:]
    for i in range(len(time_step)):
        state_history.append([time_step[i]] + list((his_o_o[i], his_o_o[i + 1], his_o_o[i + 2],
                                                    his_s_s[i], his_s_s[i + 1], his_s_s[i + 2],) + rvs))
    for i in range(3, len(his_s_s)):
        action_history.append([his_s_s[i] - his_s_s[i - 1]])

    return state_history, action_history, False


def simple_cal_reward(idx, neg, session, state_history, action_history):
    if idx == len(action_history) - 1:
        if session.agreement is None:
            reward = neg.reserved_value
            # reward = -1
        else:
            reward = neg.ufun(session.agreement)
        done = 1
    else:
        reward = 0
        done = 0

    return reward, done


def cal_reward(episode, idx, neg, session, state_history, action_history):
    upper = random.uniform(0.0, max(2 - 0.02 * (episode // 1000), 0.01))
    if idx == len(action_history) - 1:
        # if session.agreement is None:
        #     if neg.ufun(session.agreement) < neg.nash_points()[0]:
        #         if random.random() < 0.1:
        #             reward = neg.reserved_value
        #         else:
        #             reward = -10
        #     else:
        #         reward = neg.reserved_value
        #     reward = neg.reserved_value
        #     reward = -1
        # else:
        #     reward = neg.ufun(session.agreement)
        #     reward = (neg.ufun(session.agreement) - neg.reserved_value / neg.ufun.max() - neg.reserved_value)
        #     if neg.ufun(session.agreement) >= neg.nash_points()[0]:
        #         reward += 10
        if neg.ufun(session.agreement) < neg.nash_points()[0] - upper:
            reward = -1
        else:
            reward = neg.ufun(session.agreement)
        done = 1
    else:
        reward = 0
        if action_history[idx][0] < 0:
            if state_history[idx][6] < neg.nash_points()[0] - upper:
                if state_history[idx][0] == 0:
                    reward -= 0.01
            elif state_history[idx][6] + action_history[idx][0] < neg.nash_points()[0] - upper:
                if random.random() < 0.75:
                    reward -= 0.01

        done = 0

    return reward, done


def get_random(value):
    _value = value
    value = value + random.uniform(-5e-3, 5e-3)
    if value < 0.0 or value > 1.0:
        value = _value
    return value


def random_state(state):
    return (state[0],
            state[1],
            state[2],
            state[3],
            state[4],
            state[5],
            state[6],
            state[7],
            get_random(state[8]),
            )


def random_buffer(buffer):
    return (random_state(buffer[0]), buffer[1], buffer[2], random_state(buffer[3]), buffer[4])


def get_reward(episode, session, neg, Buffer, state_history, action_history):
    rewards = []

    for idx in range(len(action_history)):
        # reward, done = simple_cal_reward(idx, neg, session, state_history, action_history)
        reward, done = cal_reward(episode, idx, neg, session, state_history, action_history)

        buffer = (state_history[idx], action_history[idx], reward, state_history[idx + 1], float(done))
        buffer = random_buffer(buffer)

        if done:
            Buffer.add(buffer)
            rewards.append(reward)
        else:
            # if reward != 0 or (reward == 0 and random.random() < 0.1 / (episode // 100 + 1)):
            if random.random() < 0.01:
                Buffer.add(buffer)
                rewards.append(reward)
    return rewards


def distance_to_nash(session, pos):
    nash_point = session.nash_points()[0][0]

    if 0 in pos:
        neg = session.get_negotiator(session.negotiator_ids[0])
        # distance = math.sqrt((neg.ufun(session.agreement) - nash_point[0]) ** 2 +
        #                      (neg.opponent_ufun(session.agreement) - nash_point[1]) ** 2)
    else:
        neg = session.get_negotiator(session.negotiator_ids[1])
        # distance = math.sqrt((neg.ufun(session.agreement) - nash_point[1]) ** 2 +
        #                      (neg.opponent_ufun(session.agreement) - nash_point[0]) ** 2)
    distance = neg.ufun(session.agreement) - neg.nash_points()[0]
    # assert distance == _distance
    return distance


def save_buffer_data(Buffer, file_path):
    data = []
    for buf in Buffer.buffer:
        data.append(list(buf[0]))
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def train(SACagent, Buffer, start_episodes, num_episodes, minimal_size, batch_size, update_interval, test_interval, save_interval):
    record_rw = []
    record_dist = []
    rewards = []
    distances = []
    distances_fail = []
    best_rank = 100
    new_opponent = random.choice(TRAIN_COMPETITORS)

    for episode in trange(start_episodes, num_episodes+start_episodes):
        if episode % 10 == 0:
            new_opponent = random.choice(TRAIN_COMPETITORS)
            logging.warning(f"CHANGING opponent. New: {new_opponent.__name__}")

        # session, pos = inital_session(new_opponent, SACagent)
        pool = [(episode // 1000 - i) * 1000 for i in range(5)]
        session, pos = inital_session(new_opponent, SACagent, f"./checkpoints/model_{random.choice(pool)}.pth")
        session.run()

        distance = distance_to_nash(session, pos)

        neg = session.get_negotiator(session.negotiator_ids[0]) if pos == [0] else (
            session.get_negotiator(session.negotiator_ids[1]))
        # logging.warning(f"TIME: {session.state.relative_time:.2f} || "
        #                 f"BID: ({neg1.ufun(session.state.agreement):.2f}, "
        #                 f"{neg2.ufun(session.state.agreement):.2f}) ||"
        #                 f"NASH:{distance:.2f}")

        state_history, action_history, error = reformat_history(session, neg)

        if error:
            logging.warning("An ERROR occur!!!")
            continue

        rewards = get_reward(episode, session, neg, Buffer, state_history, action_history)
        rewards.extend(rewards)


        if session.state.agreement is None:
            distances_fail.append(distance)
        else:
            distances.append(distance)

        if Buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = Buffer.sample(batch_size)
            # logging.warning(f"zero rate: {b_r.count(0) / len(b_r)}")
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
            SACagent.update(transition_dict)

        if episode % update_interval == 0:
            _rw = sum(rewards)
            logging.warning(f"Reward: {_rw:.3f}")
            rewards = []
            record_rw.append(_rw)

            logging.warning(f"Success % = {len(distances) / (len(distances) + len(distances_fail)):.3f}")
            if distances:
                _dis = sum(distances) / len(distances)
                logging.warning(f"Success Distance to Nash: {_dis:.3f}")
            else:
                logging.warning(f"Success Distance to Nash: No")
            if distances_fail:
                _dis = sum(distances_fail) / len(distances_fail)
                logging.warning(f"Fail Distance to Nash: {_dis:.3f}")
            else:
                logging.warning(f"Fail Distance to Nash: No")
            _dis = (sum(distances) + sum(distances_fail)) / (len(distances) + len(distances_fail))
            logging.warning(f"Overall Distance to Nash: {_dis}")
            distances = []
            distances_fail = []
            record_dist.append(_dis)

            session.plot(ylimits=(0.0, 1.01), show_reserved=False, mark_max_welfare_points=False)
            plt.savefig(f'figs/episode_{episode}_pos_{pos[0]}.png')
            plt.close()
            plt.plot([i * update_interval for i in range(len(record_rw))], record_rw, marker='o')
            plt.savefig('record_rewards.png')
            plt.close()
            plt.plot([i * update_interval for i in range(len(record_dist))], record_dist, marker='o')
            plt.savefig('record_distances.png')
            plt.close()
            # logging.warning("figure saved!!!")

        if episode % save_interval == 0 and episode != 0:
            torch.save(SACagent, f'./checkpoints/model_{episode}.pth')

        if episode % test_interval == 0 and episode != 0:
            # logging.warning("Now TESTING...")
            result = run_a_tournament(partial(MyNegotiator, mode="test", model_path=SACagent),
                                      n_repetitions=1,
                                      n_outcomes=1000,
                                      n_scenarios=2,
                                      small=False, debug=False, nologs=True)
            rank = result.index[result['strategy'] == 'MyNegotiator'].tolist()[0]

            torch.save(SACagent, 'model.pth')
            logging.warning("SAVING model.pth")
            if rank <= best_rank:
                torch.save(SACagent, 'best_model.pth')
                logging.warning("SAVING best_model.pth")
                best_rank = rank

            # assert False

    # save_buffer_data(Buffer, "buffer_record.pkl")



if __name__ == "__main__":
    if os.path.exists('./figs'):
        shutil.rmtree('./figs')
    os.makedirs('./figs')

    if Path('model.pth').exists():
        os.remove('model.pth')

    if Path('best_model.pth').exists():
        os.remove('best_model.pth')

    if Path('model_vs_bot.pth').exists():
        os.remove('model_vs_bot.pth')


    Buffer = ReplayBuffer(capacity=100000)

    start_episodes = 18000
    num_episodes = 10000
    minimal_size = 1000
    batch_size = 256
    update_interval = 20
    test_interval = 1000
    save_interval = 100

    if start_episodes != 0:
        logging.warning("Loading existing checkpoint...")
        agent = torch.load(f"./checkpoints/model_{start_episodes}.pth")
    else:
        agent = SACContinuous()

    train(agent, Buffer, start_episodes, num_episodes, minimal_size, batch_size, update_interval, test_interval, save_interval)