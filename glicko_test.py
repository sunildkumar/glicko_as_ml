# this file tests my glicko implementation against the examples in the paper
from glicko import Player
import unittest


class TestGlicko(unittest.TestCase):
    def test_basics(self):
        """
        This approximates the example in the paper, and tests the basic intermediate functions of the glicko system
        """
        p = Player(rating=1500, rd=200)

        opponent1 = Player(rating=1400, rd=30)
        self.assertAlmostEqual(Player.g(opponent1.rd), 0.9955, places=4)

        expectation = p.E(opponent1)
        self.assertAlmostEqual(expectation, 0.639, places=3)

        opponent2 = Player(rating=1550, rd=100)
        opponent3 = Player(rating=1700, rd=300)

        p.update(opponent1, 1)
        p.update(opponent2, 0)
        p.update(opponent3, 0)

        self.assertAlmostEqual(p.rating, 1464, places=0)
        self.assertAlmostEqual(p.rd, 151, places=0)


if __name__ == "__main__":
    unittest.main()
