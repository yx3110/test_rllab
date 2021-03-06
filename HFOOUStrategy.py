import numpy as np
import numpy.random as nr
from rllab.core.serializable import Serializable
from rllab.exploration_strategies.base import ExplorationStrategy
from rllab.misc.overrides import overrides

{}


class HFOOUStrategy(ExplorationStrategy, Serializable):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """

    def __init__(self, env_spec, mu=0, theta=0.15, sigma=0.3, **kwargs):
        Serializable.quick_init(self, locals())
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = env_spec.action_space
        self.state = np.ones(self.action_space.flat_dim) * self.mu
        self.reset()

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["state"] = self.state
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.state = d["state"]

    @overrides
    def reset(self):
        self.state = np.ones(self.action_space.flat_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        print (self.state)
        return self.state

    @overrides
    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        ou_state = self.evolve_state()
        print(action)
        for i in range(len(action)):
            if i <= 2:
                np.clip(action[i] + ou_state, -1, 1)
            if i == 3 or i == 6:
                np.clip(action[i] + ou_state, 0, 100)
            else:
                np.clip(action[i] + ou_state, -180, 180)
        return action
