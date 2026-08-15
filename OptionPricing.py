import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

N = stats.norm.cdf
n = stats.norm.pdf


class BlackScholes:

    def __init__(self, S, K, t, sigma, r=0.05):
        self.S = S
        self.K = K
        self.t = t / 365
        self.sigma = sigma
        self.r = r

    def _d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.t) \
            / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)
        return d1, d2

    def call_price(self):
        d1, d2 = self._d1_d2()
        return N(d1) * self.S - N(d2) * self.K * np.exp(-self.r * self.t)

    def put_price(self):
        d1, d2 = self._d1_d2()
        return N(-d2) * self.K * np.exp(-self.r * self.t) - N(-d1) * self.S

    def greeks(self):
        d1, d2 = self._d1_d2()
        sqrt_t = np.sqrt(self.t)
        disc = np.exp(-self.r * self.t)

        gamma = n(d1) / (self.S * self.sigma * sqrt_t)
        vega = self.S * n(d1) * sqrt_t

        call_theta = (-(self.S * n(d1) * self.sigma) / (2 * sqrt_t)
                      - self.r * self.K * disc * N(d2))
        put_theta = (-(self.S * n(d1) * self.sigma) / (2 * sqrt_t)
                     + self.r * self.K * disc * N(-d2))

        return {
            "call": {
                "delta": N(d1),
                "gamma": gamma,
                "vega": vega / 100,
                "theta": call_theta / 365,
                "rho": self.K * self.t * disc * N(d2) / 100,
            },
            "put": {
                "delta": N(d1) - 1,
                "gamma": gamma,
                "vega": vega / 100,
                "theta": put_theta / 365,
                "rho": -self.K * self.t * disc * N(-d2) / 100,
            },
        }


class MonteCarlo:

    def __init__(self, S, K, t, sigma, simulations, r=0.05):
        self.S = S
        self.K = K
        self.t = t / 365
        self.sigma = sigma
        self.r = r
        self.N = int(simulations)
        self.num_steps = int(t)
        self.dt = self.t / self.num_steps
        self.paths = None

    def simulate_prices(self, seed=20):
        np.random.seed(seed)
        S = np.zeros((self.num_steps, self.N))
        S[0] = self.S
        for step in range(1, self.num_steps):
            Z = np.random.standard_normal(self.N)
            S[step] = S[step - 1] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * self.dt
                + self.sigma * np.sqrt(self.dt) * Z
            )
        self.paths = S
        return S

    def call_price(self):
        if self.paths is None:
            self.simulate_prices()
        payoff = np.maximum(self.paths[-1] - self.K, 0)
        return np.exp(-self.r * self.t) * payoff.mean()

    def put_price(self):
        if self.paths is None:
            self.simulate_prices()
        payoff = np.maximum(self.K - self.paths[-1], 0)
        return np.exp(-self.r * self.t) * payoff.mean()

    def plot_paths(self, num_of_movements=50):
        if self.paths is None:
            self.simulate_prices()
        num_of_movements = min(num_of_movements, self.N)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.paths[:, :num_of_movements], linewidth=0.8)
        ax.axhline(self.K, color="k", linestyle="--", label="Strike price")
        ax.set_xlim(0, self.num_steps)
        ax.set_xlabel("Days in the future")
        ax.set_ylabel("Simulated price")
        ax.set_title(f"First {num_of_movements} / {self.N} simulated price paths")
        ax.legend(loc="best")
        return fig


class BinomialTree:

    def __init__(self, S, K, t, sigma, steps, r=0.05):
        self.S = S
        self.K = K
        self.t = t / 365
        self.sigma = sigma
        self.r = r
        self.num_steps = int(steps)

    def _price(self, payoff_fn):
        dT = self.t / self.num_steps
        u = np.exp(self.sigma * np.sqrt(dT))
        d = 1.0 / u
        a = np.exp(self.r * dT)
        p = (a - d) / (u - d)
        q = 1.0 - p

        S_T = np.array([self.S * u ** j * d ** (self.num_steps - j)
                        for j in range(self.num_steps + 1)])
        V = payoff_fn(S_T)

        for _ in range(self.num_steps):
            V[:-1] = np.exp(-self.r * dT) * (p * V[1:] + q * V[:-1])
        return V[0]

    def call_price(self):
        return self._price(lambda S_T: np.maximum(S_T - self.K, 0))

    def put_price(self):
        return self._price(lambda S_T: np.maximum(self.K - S_T, 0))
