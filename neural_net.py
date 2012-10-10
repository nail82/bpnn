"""
This module and class serve as the public
interface to the neural network.
"""

class NeuralNet:
    def __init__(self, squash, squash_prime):
        self.sq = squash
        self.sqp = squash_prime

    def train(self, input_pattern, teacher):
        """Presents a pattern to the network and runs
        a forward and backward (weight adjustment) pass
        on the network.

        Args:
          input_pattern: A numpy vector to present to
              the network.  Dimensionality must match the
              network input dimensionality.

          teacher: A numpy vector that is the expected
              network output for the input pattern.

        Return:
          The sum of
        """
