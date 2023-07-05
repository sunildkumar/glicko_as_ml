# this file implements the glicko rating system
# see http://www.glicko.net/glicko/glicko.pdf
import math


class Player:
    # constants
    q = math.log(10) / 400

    def __init__(self, rating=1500, rd=350):
        self.rating = rating
        self.rd = rd

    @staticmethod
    def g(RD: float):
        return 1 / math.sqrt(1 + 3 * Player.q**2 * RD**2 / math.pi**2)

    def E(self, other: "Player"):
        return 1 / (
            1 + 10 ** (-Player.g(other.rd) * (self.rating - other.rating) / 400)
        )

    def d2(self, other: "Player", s: float):
        expectation = self.E(other)

        return (
            Player.q**2 * Player.g(other.rd) ** 2 * expectation * (1 - expectation)
        ) ** -1

    def _get_updated_r(self, other: "Player", s: float):
        return self.rating + (
            Player.q / ((1 / self.rd**2) + (1 / self.d2(other, s)))
        ) * Player.g(other.rd) * (s - self.E(other))

    def _get_updated_rd(self, other: "Player", s: float):
        return max(math.sqrt(((1 / self.rd**2) + (1 / self.d2(other, s))) ** -1), 30)

    def update(self, other: "Player", s: float):
        new_r = self._get_updated_r(other, s)
        new_rd = self._get_updated_rd(other, s)

        self.rating = new_r
        self.rd = new_rd

    @staticmethod
    def expected_outcome(a: "Player", b: "Player"):
        # black decided to format this in a way that makes it hard to read so I'm turning off formatting
        # fmt: off
        return 1 / (1+ (10** (-Player.g(math.sqrt(a.rd**2 + b.rd**2))* (a.rating - b.rating)/400)))
        # fmt: on
