import math
import os

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable

from lib.helpers import NegatedBrownian

try:
    import torch
    import torchsde
except ImportError as exc:  # pragma: no cover - library is required for this simulation path
    raise ImportError("This script requires torch and torchsde for SDE simulation.") from exc

RISK_FREE_RATE = 0.06
S0 = 1.0
DT = 1.0 / 252.0
DAYS_PER_WEEK = 5
HISTORY_WEEKS = 30
FUTURE_WEEKS = 10
N_PATHS = 10_000
SEED = 0
MONEYNESS = np.linspace(0.8, 1.2, 21)

MU_RANGE = (-0.05, 0.15)
KAPPA_RANGE = (0.5, 5.0)
THETA_RANGE = (0.02, 0.09)
SIGMA_RANGE = (0.10, 0.60)
RHO_RANGE = (-0.90, 0.30)

MC_SAMPLES = 2000        # MC samples per path (antithetic pairs => 4000 effective draws)
MC_STEPS = 25
MC_PATH_CHUNK = 250      # paths priced per batched sdeint call (controls memory)


def _set_torch_seed(seed=None):
    if seed is not None:
        torch.manual_seed(seed)


class HestonSDE(torch.nn.Module):
    """P-measure Heston SDE for state y = (log S, v), full-truncation Euler."""
    sde_type = 'ito'
    noise_type = 'general'

    def __init__(self, mu, kappa, theta, sigma, rho):
        super().__init__()
        self.mu = torch.as_tensor(mu, dtype=torch.float64)
        self.kappa = torch.as_tensor(kappa, dtype=torch.float64)
        self.theta = torch.as_tensor(theta, dtype=torch.float64)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float64)
        self.rho = torch.as_tensor(rho, dtype=torch.float64)

    def f(self, t, y):
        v = torch.clamp(y[:, 1], min=0.0)
        return torch.stack([
            self.mu - 0.5 * v,
            self.kappa * (self.theta - v),
        ], dim=1)

    def g(self, t, y):
        v = torch.clamp(y[:, 1], min=1e-12)
        sqrt_v = torch.sqrt(v)
        rho_c = torch.sqrt(1.0 - self.rho ** 2)
        diffusion = torch.zeros((y.shape[0], 2, 2), dtype=y.dtype, device=y.device)
        diffusion[:, 0, 0] = sqrt_v
        diffusion[:, 1, 0] = self.sigma * sqrt_v * self.rho
        diffusion[:, 1, 1] = self.sigma * sqrt_v * rho_c
        return diffusion


class HestonPricingSDE(torch.nn.Module):
    """Q-measure Heston SDE used for option pricing.

    Parameters may be scalars or per-sample vectors of length y.shape[0].
    """
    sde_type = 'ito'
    noise_type = 'general'

    def __init__(self, r, kappa, theta, sigma, rho):
        super().__init__()
        self.r = torch.as_tensor(r, dtype=torch.float64)
        self.kappa = torch.as_tensor(kappa, dtype=torch.float64)
        self.theta = torch.as_tensor(theta, dtype=torch.float64)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float64)
        self.rho = torch.as_tensor(rho, dtype=torch.float64)

    def f(self, t, y):
        v = torch.clamp(y[:, 1], min=0.0)
        return torch.stack([
            self.r - 0.5 * v,
            self.kappa * (self.theta - v),
        ], dim=1)

    def g(self, t, y):
        v = torch.clamp(y[:, 1], min=1e-12)
        sqrt_v = torch.sqrt(v)
        rho_c = torch.sqrt(1.0 - self.rho ** 2)
        diffusion = torch.zeros((y.shape[0], 2, 2), dtype=y.dtype, device=y.device)
        diffusion[:, 0, 0] = sqrt_v
        diffusion[:, 1, 0] = self.sigma * sqrt_v * self.rho
        diffusion[:, 1, 1] = self.sigma * sqrt_v * rho_c
        return diffusion


def sample_parameters(n, seed=SEED):
    rng = np.random.default_rng(seed)
    kappa = rng.uniform(KAPPA_RANGE[0], KAPPA_RANGE[1], size=n)
    theta = rng.uniform(THETA_RANGE[0], THETA_RANGE[1], size=n)
    sigma = rng.uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], size=n)
    sigma = _enforce_feller(kappa, theta, sigma)
    return {
        "mu": rng.uniform(MU_RANGE[0], MU_RANGE[1], size=n),
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rng.uniform(RHO_RANGE[0], RHO_RANGE[1], size=n),
    }


def simulate_heston_paths(params, n_steps, dt=DT, substeps=10, seed=SEED):
    """Simulate Heston paths under P using `torchsde`.

    Returns arrays S and V with shape (n_steps + 1, n_paths).
    """
    _set_torch_seed(seed)
    mu = torch.as_tensor(params["mu"], dtype=torch.float64)
    kappa = torch.as_tensor(params["kappa"], dtype=torch.float64)
    theta = torch.as_tensor(params["theta"], dtype=torch.float64)
    sigma = torch.as_tensor(params["sigma"], dtype=torch.float64)
    rho = torch.as_tensor(params["rho"], dtype=torch.float64)

    if not _feller_condition_satisfied(kappa.numpy(), theta.numpy(), sigma.numpy()):
        raise ValueError("Feller condition violated before simulating Heston paths.")

    n = mu.numel()
    rng = np.random.default_rng(seed + 1)
    # Stationary distribution of the CIR variance process.
    v0 = rng.gamma(
        shape=2 * kappa.numpy() * theta.numpy() / sigma.numpy() ** 2,
        scale=sigma.numpy() ** 2 / (2 * kappa.numpy()),
    )

    total_time = n_steps * dt
    ts = torch.linspace(0.0, total_time, steps=n_steps + 1, dtype=torch.float64)
    y0 = torch.stack([
        torch.full((n,), np.log(S0), dtype=torch.float64),
        torch.as_tensor(v0, dtype=torch.float64),
    ], dim=1)

    sde = HestonSDE(mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    ys = torchsde.sdeint(
        sde,
        y0=y0,
        ts=ts,
        method='euler',
        dt=dt / substeps,
    )

    # ys has shape (n_steps + 1, n_paths, 2): index the last (state) dimension.
    S = torch.exp(ys[:, :, 0]).numpy()
    V = ys[:, :, 1].numpy()
    assert S.shape == (n_steps + 1, n), f"unexpected S shape {S.shape}"
    return S, np.maximum(V, 0.0)


def monte_carlo_heston_calls(spot, var, kappa, theta, sigma, rho, r, T, strikes,
                             n_samples=MC_SAMPLES, n_steps=MC_STEPS,
                             seed=SEED, path_chunk=MC_PATH_CHUNK):
    """European call prices under Q with antithetic sampling via `NegatedBrownian`.

    Paths are batched: each chunk of `path_chunk` paths is simulated in a single
    sdeint call with `path_chunk * n_samples` independent samples.
    """
    spot = np.asarray(spot, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)

    if not _feller_condition_satisfied(kappa, theta, sigma):
        raise ValueError("Feller condition violated before option pricing.")

    n_paths = len(spot)
    n_strikes = strikes.shape[-1]
    prices = np.empty((n_paths, n_strikes), dtype=np.float64)
    discount = math.exp(-float(r) * float(T))
    dt = float(T) / n_steps
    ts = torch.linspace(0.0, float(T), steps=n_steps + 1, dtype=torch.float64)

    def rep(x):
        """Repeat each path's value n_samples times -> shape (chunk * n_samples,)."""
        return torch.as_tensor(x, dtype=torch.float64).repeat_interleave(n_samples)

    for start in tqdm(range(0, n_paths, path_chunk), desc="Pricing paths", unit="chunk"):
        stop = min(start + path_chunk, n_paths)
        m = stop - start
        _set_torch_seed(seed + start)

        y0 = torch.stack([
            rep(np.log(spot[start:stop])),
            rep(var[start:stop]),
        ], dim=1)

        sde = HestonPricingSDE(
            r=float(r),
            kappa=rep(kappa[start:stop]),
            theta=rep(theta[start:stop]),
            sigma=rep(sigma[start:stop]),
            rho=rep(rho[start:stop]),
        )
        bm = torchsde.BrownianInterval(
            t0=ts[0],
            t1=ts[-1],
            size=(m * n_samples, 2),
            dtype=torch.float64,
            device=torch.device('cpu'),
            dt=dt,
        )
        ys = torchsde.sdeint(sde, y0=y0, ts=ts, method='euler', dt=dt, bm=bm)
        ys_anti = torchsde.sdeint(sde, y0=y0, ts=ts, method='euler', dt=dt, bm=NegatedBrownian(bm))

        S_T = torch.exp(ys[-1, :, 0]).reshape(m, n_samples)
        S_T_anti = torch.exp(ys_anti[-1, :, 0]).reshape(m, n_samples)

        K = torch.as_tensor(strikes[start:stop], dtype=torch.float64)[:, None, :]  # (m, 1, n_strikes)
        payoff = torch.clamp(S_T[:, :, None] - K, min=0.0)
        payoff_anti = torch.clamp(S_T_anti[:, :, None] - K, min=0.0)
        combined = 0.5 * (payoff + payoff_anti)
        prices[start:stop] = discount * combined.mean(dim=1).numpy()

    return prices


def build_dataset(
    n_paths=N_PATHS,
    history_weeks=HISTORY_WEEKS,
    future_weeks=FUTURE_WEEKS,
    dt=DT,
    seed=SEED,
    r=RISK_FREE_RATE,
    moneyness=MONEYNESS,
    mc_samples=MC_SAMPLES,
    mc_steps=MC_STEPS,
):
    history_days = history_weeks * DAYS_PER_WEEK
    future_days = future_weeks * DAYS_PER_WEEK
    history_len = history_days + 1
    future_len = future_days
    total_steps = history_days + future_days

    params = sample_parameters(n_paths, seed=seed)
    S, V = simulate_heston_paths(params, total_steps, dt=dt, substeps=10, seed=seed)

    rng = np.random.default_rng(seed + 2)
    order = rng.permutation(n_paths)
    n_tr = int(round(0.8 * n_paths))
    n_va = int(round(0.1 * n_paths))
    split = np.empty(n_paths, dtype=np.int8)
    split[order[:n_tr]] = 0
    split[order[n_tr : n_tr + n_va]] = 1
    split[order[n_tr + n_va :]] = 2

    hist_S = np.ascontiguousarray(S[:history_len].T)
    fut_S = np.ascontiguousarray(S[history_len:].T)
    hist_V = np.ascontiguousarray(V[:history_len].T)
    fut_V = np.ascontiguousarray(V[history_len:].T)

    spot = hist_S[:, -1]
    var = hist_V[:, -1]
    maturity = future_len * dt
    strikes = spot[:, None] / np.asarray(moneyness, dtype=float)[None, :]
    option_prices = monte_carlo_heston_calls(
        spot,
        var,
        params["kappa"],
        params["theta"],
        params["sigma"],
        params["rho"],
        r,
        maturity,
        strikes,
        n_samples=mc_samples,
        n_steps=mc_steps,
        seed=seed + 10,
    )

    data = {
        "path_id": np.arange(n_paths),
        "split": split,
        "drift": params["mu"],
        "kappa": params["kappa"],
        "theta": params["theta"],
        "sigma": params["sigma"],
        "rho": params["rho"],
        "r": np.full(n_paths, float(r)),
        "spot_price": spot,
        "variance": var,
        "history_S": hist_S,
        "history_V": hist_V,
        "future_S": fut_S,
        "future_V": fut_V,
        "option_prices": option_prices,
        "moneyness": np.asarray(moneyness, dtype=float),
        "strikes": strikes,
    }
    return data


def save_csv(data, path):
    """Write to CSV."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    join = lambda arr: [",".join(map(repr, row)) for row in np.asarray(arr).tolist()]

    df = pd.DataFrame({
        "path_id": data["path_id"],
        "split": data["split"],
        "drift": data["drift"],
        "kappa": data["kappa"],
        "theta": data["theta"],
        "sigma": data["sigma"],
        "rho": data["rho"],
        "r": data["r"],
        "spot_price": data["spot_price"],
        "variance": data["variance"],
        "history_S": join(data["history_S"]),
        "history_V": join(data["history_V"]),
        "future_S": join(data["future_S"]),
        "future_V": join(data["future_V"]),
    })

    for i, mny in enumerate(data["moneyness"]):
        df[f"option_price_moneyness={mny:.3f}"] = data["option_prices"][:, i]

    df.to_csv(path, index=False)
    return path


def _feller_condition_satisfied(kappa, theta, sigma, margin=1e-12):
    """Return True iff 2*kappa*theta >= sigma^2 for all parameter sets.

    Use a small relative tolerance so values that are numerically equal up to
    floating-point rounding are accepted instead of being rejected as invalid.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    gap = 2.0 * kappa * theta - sigma ** 2
    tol = np.maximum(np.abs(gap), 0.0) * 0.0 + margin * np.maximum(1.0, np.maximum(2.0 * kappa * theta, sigma ** 2))
    return np.all((kappa > 0.0) & (theta > 0.0) & (sigma >= 0.0) & (gap >= -tol))


def _enforce_feller(kappa, theta, sigma):
    """Clip sigma so every parameter set satisfies the Feller condition."""
    kappa = np.asarray(kappa, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if not np.all((kappa > 0) & (theta > 0)):
        raise ValueError("kappa and theta must be strictly positive to satisfy Feller.")

    cap = np.sqrt(2.0 * kappa * theta)
    sigma = np.minimum(sigma, np.nextafter(cap, 0.0))
    if not _feller_condition_satisfied(kappa, theta, sigma):
        raise ValueError("Some parameter sets violate the Feller condition: 2*kappa*theta >= sigma^2.")
    return sigma


if __name__ == "__main__":
    dataset = build_dataset()
    out = save_csv(dataset, "datasets/heston.csv")
    counts = [int((dataset["split"] == i).sum()) for i in range(3)]
    print(f"{len(dataset['path_id'])} paths | train/val/test rows = {counts}")
    print(f"history: {dataset['history_S'].shape[1]} prices  | future: {dataset['future_S'].shape[1]} prices")
    print(f"saved {out}")