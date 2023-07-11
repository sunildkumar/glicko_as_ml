# This is an extension of glicko_experiment1.py. Instead of players having a fixed skill against all games, their skill varies from game to game. This is a more realistic model of the world as some players are better at different games.
# Random idea: represent a player's hidden skill/a game's hidden difficulty as a random vector with length one. Then the win rate is a function of the dot product of the two vectors. This would allow for players to be better at some games than others.

from typing import Tuple
from scipy.stats import truncnorm
from glicko import Player as GlickoPlayer
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from tqdm import tqdm

random.seed(42)
np.random.seed(42)


def draw_samples(mean=0.5, sd=1.0, low=0.0, upp=1.0, size=1):
    """
    Return a sequence of numbers between low and upp drawn from a truncated normal distribution with given mean and standard deviation.
    """
    a = (low - mean) / sd
    b = (upp - mean) / sd
    samples = truncnorm(a, b, loc=mean, scale=sd).rvs(size)
    return samples


# extend the glicko player class for this experiment
# a player will have a different hidden skill for each game they play
class Player(GlickoPlayer):
    def __init__(self, num_games):
        super().__init__()
        self.num_games = num_games
        # hidden_skill_mean is modeling the true skill of the player
        self.hidden_skill_mean = draw_samples()[0]
        # hidden_skill_sd is modeling the variance in the player's skill across games
        self.hidden_skill_sd = random.uniform(0.01, 0.2)

        # the players skill against each game
        self.hidden_game_skills = draw_samples(
            mean=self.hidden_skill_mean, sd=self.hidden_skill_sd, size=num_games
        )

    def get_hidden_game_skill(self, game_index: int) -> float:
        """
        returns the skill of the player at the game at game_index
        """
        return self.hidden_game_skills[game_index]


# a game has a hidden difficulty that is the same for all players who play it
class Game(GlickoPlayer):
    def __init__(self):
        super().__init__()
        # hidden_difficulty is modeling the true difficulty of the game
        self.hidden_difficulty = draw_samples()[0]

    def get_hidden_difficulty(self) -> float:
        return self.hidden_difficulty


def win_rate_model(player_skill: float, game_difficulty: float) -> float:
    """
    Returns the likelihood that a player with skill player_skill will win against a game with difficulty game_difficulty.
    """
    probability = 1 / (
        1
        + np.exp(-(15 * (player_skill**2 - 0.5) - 10 * (game_difficulty**2 - 0.5)))
    )
    assert 0 <= probability <= 1
    return probability


def win_rate_lookup_table(players: list[Player], games: list[Game]) -> np.ndarray:
    """
    Precomputes the win rate of each player against each game. Returns a lookup table of the form lookup_table[game_index, player_index] = win_rate(player, game)
    """
    true_win_probs = np.zeros((len(games), len(players)))
    for player_index, player in enumerate(players):
        for game_index, game in enumerate(games):
            win_prob = win_rate_model(
                player.get_hidden_game_skill(game_index), game.get_hidden_difficulty()
            )
            true_win_probs[game_index, player_index] = win_prob

    return true_win_probs


def find_winner(
    player_index: int,
    game_index: int,
    lookup_table: np.ndarray,
) -> Tuple[int, int]:
    """
    Returns a tuple of the form (player_wins, game_wins)where
    player_wins is 1 if the player wins and 0 if the player
    loses and game_wins is 1 if the game wins and 0 if the game loses.
    """
    win_probability = lookup_table[game_index, player_index]
    player_wins = int(win_probability > random.random())
    game_wins = 1 - player_wins
    return player_wins, game_wins


def play_game(
    player: Player,
    game: Game,
    player_index: int,
    game_index: int,
    lookup_table: np.ndarray,
):
    """
    Simulates a game between player and game. Updates the player and game ratings accordingly.
    """
    player_wins, game_wins = find_winner(player_index, game_index, lookup_table)
    player.update(game, player_wins)
    game.update(player, game_wins)


def create_players_and_games(num_players, num_games):
    """
    Returns a tuple of the form (players, games) where players is a list of num_players players and games is a list of num_games games. The players and games are sorted by hidden skill/difficulty.
    """
    players = [Player(num_games) for _ in range(num_players)]
    games = [Game() for _ in range(num_games)]
    players.sort(key=lambda x: x.hidden_skill_mean)
    games.sort(key=lambda x: -x.get_hidden_difficulty())
    return players, games


def create_true_model_figure(
    players: list[Player], games: list[Game], true_win_probs: np.ndarray
):
    """
    Creates a figure that shows the true model of the world. The figure shows the true win rate of each player against each game.
    """

    colors = ["darkred", "red", "orange", "lime", "green"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    plt.imshow(true_win_probs, cmap=cmap, interpolation="nearest", norm=norm)
    plt.xlabel("Players organized from worst to best")
    plt.ylabel("Games organized from easiest to hardest")
    plt.colorbar()
    plt.title("True probability a player wins game")
    plt.savefig("figures2/true_win_probs.png")
    plt.close()


def make_skill_rating_figure(players: list[Player], games: list[Game], iteration):
    """
    plots the skill/difficulty of each player/game against its glicko rating
    """
    player_ratings = [player.rating for player in players]
    player_skills = [player.hidden_skill_mean for player in players]

    game_ratings = [game.rating for game in games]
    game_difficulties = [game.get_hidden_difficulty() for game in games]

    players_2RD = [2 * player.rd for player in players]
    games_2RD = [2 * game.rd for game in games]

    plt.scatter(
        player_ratings,
        player_skills,
    )
    plt.scatter(game_ratings, game_difficulties)
    errorbar_kwargs = dict(
        fmt="o", capsize=5, markersize=6, markeredgewidth=1.5, elinewidth=1.5
    )
    plt.errorbar(
        player_ratings,
        player_skills,
        xerr=players_2RD,
        label="Players 95% confidence interval",
        **errorbar_kwargs,
    )
    plt.errorbar(
        game_ratings,
        game_difficulties,
        xerr=games_2RD,
        label="Games 95% confidence interval",
        **errorbar_kwargs,
    )
    plt.xlabel("Glicko Rating")
    plt.ylabel("Hidden skill/difficulty")
    plt.legend()
    plt.title(f"Rounds: {iteration}")
    plt.savefig(f"figures2/{iteration}.png")
    plt.close()


def predicted_win_rate_table(players: list[Player], games: list[Game]) -> np.ndarray:
    """
    given a list of players and games, returns a table of the form predicted_win_probs[game_index, player_index] = predicted_win_probability(player, game)
    """

    predicted_win_probs = np.zeros((len(games), len(players)))
    for player_index, player in enumerate(players):
        for game_index, game in enumerate(games):
            predicted_win_probability = GlickoPlayer.expected_outcome(a=player, b=game)
            predicted_win_probs[game_index, player_index] = predicted_win_probability

    return predicted_win_probs


def make_predicted_model_figure(players: list[Player], games: list[Game]):
    predicted_table = predicted_win_rate_table(players, games)

    colors = ["darkred", "red", "orange", "lime", "green"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    plt.imshow(predicted_table, cmap=cmap, interpolation="nearest", norm=norm)
    plt.xlabel("Players organized from worst to best")
    plt.ylabel("Games organized from easiest to hardest")
    plt.colorbar()
    plt.title("Estimated probability a player wins game")
    plt.savefig("figures2/predicted_win_probs.png")
    plt.close()


def mse_2d(arr1, arr2):
    """
    calculates the mean squared error between two 2d arrays elementwise
    """
    assert arr1.shape == arr2.shape, "Arrays must have the same shape"
    flat_arr1 = arr1.flatten()
    flat_arr2 = arr2.flatten()
    mse = np.mean((flat_arr1 - flat_arr2) ** 2)
    return mse


def simulation(
    players: list[Player],
    games: list[Game],
    num_iterations: int,
    lookup_table: np.ndarray,
):
    """
    Runs the simulation for num_iterations iterations. In each iteration, each player plays a random game
    """
    for iteration in tqdm(range(num_iterations)):
        # in each round each player plays a random game
        for player_index, player in enumerate(players):
            game_index = random.randint(0, len(games) - 1)
            game = games[game_index]
            play_game(player, game, player_index, game_index, lookup_table)
        if iteration % (int(0.02 * num_iterations)) == 0:
            make_skill_rating_figure(players, games, iteration)

    # save final figure
    make_skill_rating_figure(players, games, num_iterations)


players, games = create_players_and_games(num_players=1000, num_games=1000)
lookup_table = win_rate_lookup_table(players=players, games=games)
create_true_model_figure(players=players, games=games, true_win_probs=lookup_table)
simulation(players=players, games=games, num_iterations=100, lookup_table=lookup_table)
make_predicted_model_figure(players=players, games=games)

predicted_table = predicted_win_rate_table(players, games)
mse = mse_2d(lookup_table, predicted_table)
print(f"Mean squared error between true and predicted win rates: {mse}")
