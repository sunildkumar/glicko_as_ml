# This is an extension of glicko_experiment1.py. Instead of players having a fixed skill against all games, their skill varies from game to game. This is a more realistic model of the world as some players are better at different games.
# Random idea: represent a player's hidden skill/a game's hidden difficulty as a random vector with length one. Then the win rate is a function of the dot product of the two vectors. This would allow for players to be better at some games than others.

from scipy.stats import truncnorm
from glicko import Player as GlickoPlayer
import random
import numpy as np

random.seed(42)
np.random.seed(42)


def draw_sample(mean=0.5, sd=1.0, low=0.0, upp=1.0):
    """
    Return a number between low and upp drawn from a truncated gaussian with mean and sd.
    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(1)[0]


# extend the glicko player class for this experiment
# a player will have a different hidden skill for each game they play
class Player(GlickoPlayer):
    def __init__(self, num_games):
        super().__init__()
        self.num_games = num_games
        # hidden_true_skill_mean is modeling the true skill of the player
        self.hidden_true_skill_mean = draw_sample()
        # hidden_skill_sd is modeling the variance in the player's skill across games
        self.hidden_true_skill_sd = random.uniform(0.01, 0.2)

        # the players skill against each game
        self.hidden_game_skills = [
            draw_sample(mean=self.hidden_true_skill_mean, sd=self.hidden_true_skill_sd)
            for _ in range(self.num_games)
        ]

    def get_game_skill(self, game_index) -> float:
        return self.hidden_game_skills[game_index]


# a game has a hidden difficulty that is the same for all players who play it
class Game(GlickoPlayer):
    def __init__(self):
        super().__init__()
        # hidden_true_difficulty is modeling the true difficulty of the game
        self.hidden_true_difficulty = draw_sample()

    def get_true_difficulty(self) -> float:
        return self.hidden_true_difficulty


def win_rate(player_skill, game_difficulty) -> float:
    probability = 1 / (
        1
        + np.exp(-(15 * (player_skill**2 - 0.5) - 10 * (game_difficulty**2 - 0.5)))
    )
    assert 0 <= probability <= 1
    return probability
