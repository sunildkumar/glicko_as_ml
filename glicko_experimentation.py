from typing import Tuple
from glicko import Player
import random
import numpy as np
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
from tqdm import tqdm

np.random.seed(42)
random.seed(42)


# define function to draw number between 0 and 1 from truncated gaussian
def draw_sample(mean=0.5, sd=1, low=0, upp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(1)[0]


# I'm particularly interested in glicko on games where there are two distinct groups where they only play against each other but not internally. Call these groups A, B.
# All players will have identical glicko ratings to start, however the underlying skill of players will be represented explicitly as a number between 0 and 1 drawn from the gaussian.
group_a_size = 100
group_b_size = 1000
group_a = [(Player(), draw_sample()) for _ in range(group_a_size)]
group_b = [(Player(), draw_sample()) for _ in range(group_b_size)]


# Let's define the likelihood that player a from A wins against player b from B as win_rate(a, b)
def win_rate(a_skill, b_skill) -> float:
    probability = (
        1 / (1 + np.exp(-(15 * (a_skill - 0.5) - 10 * (b_skill - 0.5)))) / 2 + 0.5
    )
    assert 0 <= probability <= 1
    return probability


def find_winner(a_skill: int, b_skill: int) -> Tuple[float, float]:
    # Then given the skill of players a, b we can determine the win state for players a, b
    a_wins = int(win_rate(a_skill, b_skill) > random.random())
    b_wins = 1 - a_wins
    return a_wins, b_wins


def play_game(a_index: int, b_index: int):
    a_wins, b_wins = find_winner(group_a[a_index][1], group_b[b_index][1])
    group_a[a_index][0].update(group_b[b_index][0], a_wins)
    group_b[b_index][0].update(group_a[a_index][0], b_wins)


def make_figure(iteration):
    a_ratings = [element[0].rating for element in group_a]
    a_skills = [element[1] for element in group_a]

    b_ratings = [element[0].rating for element in group_b]
    b_skills = [element[1] for element in group_b]

    a_2RD = [2 * element[0].rd for element in group_a]
    b_2RD = [2 * element[0].rd for element in group_b]

    plt.scatter(a_ratings, a_skills, label="group a")
    plt.scatter(b_ratings, b_skills, label="group b")
    errorbar_kwargs = dict(
        fmt="o", capsize=5, markersize=6, markeredgewidth=1.5, elinewidth=1.5
    )
    plt.errorbar(a_ratings, a_skills, xerr=a_2RD, **errorbar_kwargs)
    plt.errorbar(b_ratings, b_skills, xerr=b_2RD, **errorbar_kwargs)
    plt.xlabel("rating")
    plt.ylabel("skill")
    plt.legend()
    plt.savefig(f"figures/{iteration}.png")
    plt.close()


# Let's play this game many times
num_rounds = 10000
for i in tqdm(range(num_rounds)):
    # in each round, each player in group A plays a random player in group B
    for a_index in range(len(group_a)):
        b_index = random.randint(0, len(group_b) - 1)
        play_game(a_index, b_index)

    # save progress every 2% of the way as a figure
    if i % (0.02 * num_rounds) == 0:
        make_figure(i)

# save final figure
make_figure("final")
