from typing import Tuple
from glicko import Player
import random
import numpy as np
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

np.random.seed(42)
random.seed(42)


# define function to draw number between 0 and 1 from truncated gaussian
def draw_sample(mean=0.5, sd=1, low=0, upp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(1)[0]


# I'm particularly interested in glicko on games where there are two distinct groups where they only play against each other but not internally. Call these groups A, B.
# All players will have identical glicko ratings to start, however the underlying skill of players will be represented explicitly as a number between 0 and 1 drawn from the gaussian.
group_a_size = 1000  # players
group_b_size = 1000  # games
group_a = [(Player(), draw_sample()) for _ in range(group_a_size)]
group_b = [(Player(), draw_sample()) for _ in range(group_b_size)]

# sort the groups by skill
group_a.sort(key=lambda x: x[1])
group_b.sort(key=lambda x: -x[1])


# Let's define the likelihood that player a from A wins against player b from B as win_rate(a_skill, b_skill)
def win_rate(a_skill, b_skill) -> float:
    probability = 1 / (
        1 + np.exp(-(15 * (a_skill**2 - 0.5) - 10 * (b_skill**2 - 0.5)))
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

    plt.scatter(
        a_ratings,
        a_skills,
    )
    plt.scatter(b_ratings, b_skills)
    errorbar_kwargs = dict(
        fmt="o", capsize=5, markersize=6, markeredgewidth=1.5, elinewidth=1.5
    )
    plt.errorbar(
        a_ratings,
        a_skills,
        xerr=a_2RD,
        label="Players 95% confidence interval",
        **errorbar_kwargs,
    )
    plt.errorbar(
        b_ratings,
        b_skills,
        xerr=b_2RD,
        label="Games 95% confidence interval",
        **errorbar_kwargs,
    )
    plt.xlabel("Glicko Rating")
    plt.ylabel("Hidden skill/difficulty")
    plt.legend()
    plt.title(f"Rounds: {iteration}")
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
    if i % (int(0.02 * num_rounds)) == 0:
        make_figure(i)

# save final figure
make_figure(num_rounds)

# Evaluate the results
# first calculate the true win probability for each player in group A against each player in group B
true_win_probs = np.zeros((group_b_size, group_a_size))

for a_index in range(group_a_size):
    for b_index in range(group_b_size):
        true_win_probs[b_index][a_index] = win_rate(
            group_a[a_index][1], group_b[b_index][1]
        )

colors = ["darkred", "red", "orange", "lime", "green"]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)


plt.imshow(true_win_probs, cmap=cmap, interpolation="nearest", norm=norm)
plt.xlabel("Players organized from worst to best")
plt.ylabel("Games organized from easiest to hardest")
plt.colorbar()
plt.title("True probability a player wins game")
plt.savefig("figures/true_win_probs.png")
plt.close()

# now calculate the glicko win probability for each player in group A against each player in group B
estimated_win_probs = np.zeros((group_b_size, group_a_size))
for a_index in range(group_a_size):
    for b_index in range(group_b_size):
        estimated_win_probs[b_index][a_index] = Player.expected_outcome(
            a=group_a[a_index][0], b=group_b[b_index][0]
        )

plt.imshow(estimated_win_probs, cmap=cmap, interpolation="nearest", norm=norm)
plt.xlabel("Players organized from worst to best")
plt.ylabel("Games organized from easiest to hardest")
plt.colorbar()
plt.title("Estimated probability a player wins game")
plt.savefig("figures/estimated_win_probs.png")
plt.close()

difference = np.abs(true_win_probs - estimated_win_probs)
average_difference = np.mean(difference)
plt.imshow(difference, cmap="hot", interpolation="nearest")
plt.xlabel("Players organized from worst to best")
plt.ylabel("Games organized from easiest to hardest")
plt.colorbar()
plt.title(
    f"Difference between true and estimated win probability. Average percent difference = {round(average_difference*100, 3)}%"
)
plt.savefig("figures/difference.png", bbox_inches="tight")
plt.close()
