import torch
import torchsde
import pysiglib.torch_api as pysiglib

def transform_paths(paths, flatten=False, time_aug=False, base_pt_aug=True, sig_degree=0):
    """
    Applies some useful transformations to a given collection of paths.

    Arguments:

    path: A tensor of shape (num_samples, num_time_observations, dimension)

    flatten: If set to True, flattens the paths to shape (num_samples, num_time_observations * dimension)

    time_aug: If set to True, an extra dimension is added to the path representing the time coordinate. The observations will be assumed to be observed at equispaced time points in the interval
    [0,1]

    base_pt_aug: The path will be modified to start from the origin by subtracting the starting point from each observation

    sig_degree: Int: If set to a positive integer value, creates a dataset object containing path signatures truncated to the given value. Set 0 to keep the paths themselves.

    """

    if time_aug:
        ts = torch.linspace(0,1,paths.shape[1])
        ts = ts.expand(paths.shape[0], -1).unsqueeze(2)
        paths = torch.cat((ts, paths), axis=2)

    if base_pt_aug:
        paths = paths - paths[:, 0:1, :]

    if flatten:
        paths = torch.flatten(paths, start_dim=1)

    if sig_degree > 0:
        paths = pysiglib.signature(paths, degree=sig_degree, time_aug=False)

    return paths

def visibility_transform(paths):
    """
    Applies the discrete I-visibility transformation to a given collection of paths. See https://arxiv.org/pdf/2004.04006 for details.
    """
    if len(paths.shape) < 3:
        paths = paths.unsqueeze(2)
    b, N, d = paths.shape
    transformed_path = torch.zeros(b, N+2,d+1)
    transformed_path[:,2:,:-1] = paths
    transformed_path[:,2:, -1] = 1
    transformed_path[:,1, :-1] = paths[:,0,:]
    transformed_path[:,1, -1] = 0

    return transformed_path


class LipSwish(torch.nn.Module):
    def __init__(self, is_Lip=True):
        super().__init__()
        self.is_Lip = is_Lip
    def forward(self, x):
        if self.is_Lip:
            return 0.909 * torch.nn.functional.silu(x)
        else:
            return x

class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers,is_Lip=True, tanh=False):
        super().__init__()

        if num_layers > 0:
            model = [torch.nn.Linear(in_size, mlp_size),
                    LipSwish(is_Lip=is_Lip)]
            for _ in range(num_layers - 1):
                model.append(torch.nn.Linear(mlp_size, mlp_size))
                ###################
                # LipSwish activations are useful to constrain the is_Lipschitz constant of the MLP
                ###################
                model.append(LipSwish(is_Lip=is_Lip))
            model.append(torch.nn.Linear(mlp_size, out_size))
        else:
            model = [torch.nn.Linear(in_size, out_size)]
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


"""
Some stuff for testing. 
"""

class Identical_BrownianInterval(torchsde.BrownianInterval):
    def __init__(self, t0, t1, batch_size, brownian_size, device=None, dtype=None, dt=None):
        super().__init__(t0=t0, t1=t1, size=(1, brownian_size), device=device, dtype=dtype, dt=dt)
        self._bm = None
        self._batch_size = batch_size
        self._brownian_size = brownian_size

    def __call__(self, ta, tb=None):
        self._bm = super().__call__(ta, tb)
        return self._bm.expand(self._batch_size, -1)

    @property
    def shape(self):
        return (self._batch_size, self._brownian_size)


class SDE(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'general'

    def __init__(self, func):
        super().__init__()
        self.func = func

    def f_and_g(self, t, y):
        return self.func(t, y)

class NegatedBrownian(torchsde.BaseBrownian):
    """Antithetic partner of a Brownian motion: W'(ta, tb) = -W(ta, tb)."""
    def __init__(self, base_brownian):
        super().__init__()
        self.base_brownian = base_brownian

    def __call__(self, ta, tb=None, return_U=False, return_A=False):
        if return_U or return_A:
            raise NotImplementedError("Only increments are needed by the Euler solver.")
        return -self.base_brownian(ta, tb)

    def __repr__(self):
        return f"{self.__class__.__name__}(base_brownian={self.base_brownian})"

    @property
    def dtype(self): return self.base_brownian.dtype
    @property
    def device(self): return self.base_brownian.device
    @property
    def shape(self): return self.base_brownian.shape
    @property
    def levy_area_approximation(self): return self.base_brownian.levy_area_approximation
