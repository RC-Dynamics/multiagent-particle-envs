import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import gumbel_softmax, onehot_from_logits


def onehot_from_logits(logits):
    """
    Given batch of logits, return one-hot sample 
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    return argmax_acs


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps).to(torch.device('cuda'))

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


class OrnsteinUhlenbeckNoise:
    """Add Ornstein-Uhlenbeck noise to continuous actions.

        https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process_

        Parameters
        ----------
        mu : float or ndarray, optional

            The mean towards which the Ornstein-Uhlenbeck process should revert; must be
            broadcastable with the input actions.

        sigma : positive float or ndarray, optional

            The spread of the noise of the Ornstein-Uhlenbeck process; must be
            broadcastable with the input actions.

        theta : positive float or ndarray, optional

            The (element-wise) dissipation rate of the Ornstein-Uhlenbeck process; must
            be broadcastable with the input actions.

        min_value : float or ndarray, optional

            The lower bound used for clipping the output action; must be broadcastable with the input
            actions.

        max_value : float or ndarray, optional

            The upper bound used for clipping the output action; must be broadcastable with the input
            actions.

        random_seed : int, optional

            Sets the random state to get reproducible results.
    """

    def __init__(
            self,
            mu=0.,
            sigma=1.,
            theta=0.15,
            min_value=None,
            max_value=None,
            random_seed=None
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.min_value = -1e15 if min_value is None else min_value
        self.max_value = 1e15 if max_value is None else max_value
        self.random_seed = random_seed
        self.rnd = np.random.RandomState(self.random_seed)
        self.reset()

    def reset(self):
        """Reset the Ornstein-Uhlenbeck process."""
        self._noise = None

    def __call__(self, a):
        """Add some Ornstein-Uhlenbeck to a continuous action.

            Parameters
            ----------
            a : action

                A single action

            Returns
            -------
            a_noisy : action

                An action with OU noise added
        """
        a = np.asarray(a)
        if self._noise is None:
            self._noise = np.ones_like(a) * self.mu

        white_noise = np.asarray(self.rnd.randn(*a.shape), dtype=a.dtype)
        self._noise += self.theta * \
            (self.mu - self._noise) + self.sigma * white_noise
        self._noise = np.clip(self._noise, self.min_value, self.max_value)
        return a + self._noise
