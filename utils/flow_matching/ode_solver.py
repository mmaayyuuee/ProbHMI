# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import Tensor, nn
from torchdiffeq import odeint
from abc import ABC, abstractmethod
# from geomstats.geometry.special_orthogonal import SpecialOrthogonal
# import geomstats.backend as gs  # type: ignore
from typing import Callable, Optional, Sequence, Tuple, Union


def gradient(
    output: Tensor,
    x: Tensor,
    grad_outputs: Optional[Tensor] = None,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute the gradient of the inner product of output and grad_outputs w.r.t :math:`x`.

    Args:
        output (Tensor): [N, D] Output of the function.
        x (Tensor): [N, d_1, d_2, ... ] input
        grad_outputs (Optional[Tensor]): [N, D] Gradient of outputs, if `None`,
            then will use a tensor of ones
        create_graph (bool): If True, graph of the derivative will be constructed, allowing
            to compute higher order derivative products. Defaults to False.
    Returns:
        Tensor: [N, d_1, d_2, ... ]. the gradient w.r.t x.
    """

    if grad_outputs is None:
        grad_outputs = torch.ones_like(output).detach()
    grad = torch.autograd.grad(
        output, x, grad_outputs=grad_outputs, create_graph=create_graph
    )[0]
    return grad



class ModelWrapper(ABC, nn.Module):
    """
    This class is used to wrap around another model, adding custom forward pass logic.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        r"""
        This method defines how inputs should be passed through the wrapped model.
        Here, we're assuming that the wrapped model takes both :math:`x` and :math:`t` as input,
        along with any additional keyword arguments.

        Optional things to do here:
            - check that t is in the dimensions that the model is expecting.
            - add a custom forward pass logic.
            - call the wrapped model.

        | given x, t
        | returns the model output for input x at time t, with extra information `extra`.

        Args:
            x (Tensor): input data to the model (batch_size, ...).
            t (Tensor): time (batch_size).
            **extras: additional information forwarded to the model, e.g., text condition.

        Returns:
            Tensor: model output.
        """
        return self.model(x=x, t=t, **extras)


class Solver(ABC, nn.Module):
    """Abstract base class for solvers."""

    @abstractmethod
    def sample(self, x_0: Tensor = None) -> Tensor:
        ...



class ODESolver(Solver):
    """A class to solve ordinary differential equations (ODEs) using a specified velocity model.
    
    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.
    
    Args:
        velocity_model (Union[ModelWrapper, Callable]): a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`
    """

    def __init__(self, velocity_model: Union[ModelWrapper, Callable]):
        super().__init__()
        self.velocity_model = velocity_model
        return


    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        r"""Solve the ODE with the velocity field.

        Example:

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import ODESolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)

                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return torch.ones_like(x) * 3.0 * t**2

            velocity_model = DummyModel()
            solver = ODESolver(velocity_model=velocity_model)
            x_init = torch.tensor([0.0, 0.0])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to torch.tensor([0.0, 1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tensor, Sequence[Tensor]]: The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """

        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            return self.velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}
        if method in ['dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun']:
            ode_opts = None
        
        with torch.set_grad_enabled(enable_grad):
            # Approximate ODE solution with numerical ODE solver
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )
        '''
        sol = torch.squeeze(sol, dim=-3)
        if sol.ndim == 4:
            sol = sol[None, ...]
        elif sol.ndim == 3:
            sol = sol[None, : None, ...]
        if return_intermediates:
            return sol
        else:
            return sol[:, -1:]
        '''
        if return_intermediates:
            return sol
        else:
            return sol[-1]


    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        r"""Solve for log likelihood given a target sample at :math:`t=0`.

        Works similarly to sample, but solves the ODE in reverse to compute the log-likelihood. The velocity model must be differentiable with respect to x.
        The function assumes log_p0 is the log probability of the source distribution at :math:`t=0`.

        Args:
            x_1 (Tensor): target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 (Callable[[Tensor], Tensor]): Log probability function of the source distribution.
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): If step_size is None then time discretization is set by the time grid. Must start at 1.0 and end at 0.0, otherwise the likelihood computation is not valid. Defaults to torch.tensor([1.0, 0.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Otherwise only return the final sample. Defaults to False.
            exact_divergence (bool): Whether to compute the exact divergence or use the Hutchinson estimator.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]: Samples at time_grid and log likelihood values of given x_1.
        """
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        # Fix the random projection for the Hutchinson divergence estimator
        if not exact_divergence:
            z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt.requires_grad_()
                ut = ode_func(xt, t)

                if exact_divergence:
                    # # Compute exact divergence
                    # div = 0
                    # for i in range(ut.flatten(1).shape[1]):
                    #     div += gradient(ut[:, i], xt, create_graph=True)[:, i]
                    raise NotImplementedError
                else:
                    # # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                    # ut_dot_z = torch.einsum(
                    #     "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                    # )
                    # grad_ut_dot_z = gradient(ut_dot_z, xt)
                    # div = torch.einsum(
                    #     "ij,ij->i",
                    #     grad_ut_dot_z.flatten(start_dim=1),
                    #     z.flatten(start_dim=1),
                    # )
                    ut_dot_z = torch.einsum("tbij,tbij->b", ut, z)  # 输出形状: [B]                    
                    grad_ut_dot_z = gradient(ut_dot_z, xt, create_graph=True, grad_outputs=torch.ones_like(ut_dot_z))
                    div = torch.einsum("tbij,tbij->b", grad_ut_dot_z, z)
                    
            return ut.detach(), div.detach()

        # y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        y_init = (x_1, torch.zeros(x_1.shape[1], device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        else:
            return sol[-1], source_log_p + log_det[-1]



class SO3ODESolver(ODESolver):
    pass

class SO3ODESolverWithVel(SO3ODESolver):
    pass



'''  ### 弃用
class ODESolver(Solver):
    r"""A class to solve ordinary differential equations (ODEs) using a specified velocity model.
    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.
    Args:
        velocity_model : a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`
    """

    def __init__(self, velocity_model):
        super().__init__()
        self.velocity_model = velocity_model
        return 

    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        time_grid_scale: bool = True, 
        return_intermediates: bool = False,
        enable_grad: bool = False,
        hz: int = 50, 
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        r"""Solve the ODE with the velocity field.

        Example:

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import ODESolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)

                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return torch.ones_like(x) * 3.0 * t**2

            velocity_model = DummyModel()
            solver = ODESolver(velocity_model=velocity_model)
            x_init = torch.tensor([0.0, 0.0])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to torch.tensor([0.0, 1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tensor, Sequence[Tensor]]: The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """

        def ode_func(t, x):
            if not time_grid_scale:
                t = t / hz
            return self.velocity_model(x=x, t=t, **model_extras)

        time_grid = time_grid.to(x_init.device)
        if time_grid_scale:
            time_grid = time_grid / hz
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            # Approximate ODE solution with numerical ODE solver
            sol = odeint(
                func=ode_func,
                y0=x_init,
                t=time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )
        sol = torch.squeeze(sol, dim=-3)
        if sol.ndim == 4:
            sol = sol[None, ...]
        elif sol.ndim == 3:
            sol = sol[None, : None, ...]
            
        if return_intermediates:
            return sol
        else:
            return sol[:, -1:]


    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        r"""Solve for log likelihood given a target sample at :math:`t=0`.

        Works similarly to sample, but solves the ODE in reverse to compute the log-likelihood. The velocity model must be differentiable with respect to x.
        The function assumes log_p0 is the log probability of the source distribution at :math:`t=0`.

        Args:
            x_1 (Tensor): target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 (Callable[[Tensor], Tensor]): Log probability function of the source distribution.
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): If step_size is None then time discretization is set by the time grid. Must start at 1.0 and end at 0.0, otherwise the likelihood computation is not valid. Defaults to torch.tensor([1.0, 0.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Otherwise only return the final sample. Defaults to False.
            exact_divergence (bool): Whether to compute the exact divergence or use the Hutchinson estimator.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]: Samples at time_grid and log likelihood values of given x_1.
        """
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        # Fix the random projection for the Hutchinson divergence estimator
        if not exact_divergence:
            z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt.requires_grad_()
                ut = ode_func(xt, t)

                if exact_divergence:
                    # Compute exact divergence
                    div = 0
                    for i in range(ut.flatten(1).shape[1]):
                        div += gradient(ut[:, i], xt, create_graph=True)[:, i]
                else:
                    # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                    ut_dot_z = torch.einsum(
                        "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                    )
                    grad_ut_dot_z = gradient(ut_dot_z, xt)
                    div = torch.einsum(
                        "ij,ij->i",
                        grad_ut_dot_z.flatten(start_dim=1),
                        z.flatten(start_dim=1),
                    )
            return ut.detach(), div.detach()

        y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        else:
            return sol[-1], source_log_p + log_det[-1]



class SO3ODESolver(ODESolver):
    SO3 = SpecialOrthogonal(n=3, point_type="vector", equip=True)
    
    def sample(self, x_init: Tensor, step_size: Optional[float], method: str = "symplectic_euler",
               atol: float = 1e-5, rtol: float = 1e-5, time_grid: Tensor = torch.tensor([0.0, 1.0]), 
               time_grid_scale: bool = True, return_intermediates: bool = False, enable_grad: bool = False,
               hz: int = 50, **model_extras) -> Union[Tensor, Sequence[Tensor]]:

        B, _ , C, N = x_init.shape
        
        def ode_func(t, x):
            if not time_grid_scale:
                t = t / hz
            return self.velocity_model(x=x, t=t, **model_extras)
        
        if method == "symplectic_euler":
            integ_func = self.symplectic_euler_step
        elif method == "euler":
            integ_func = self.euler_step

        time_grid = time_grid.to(x_init.device)
        if time_grid_scale:
            time_grid = time_grid / hz
        steps = math.floor(time_grid[-1] / step_size)
        
        sol = torch.zeros(size=(B, steps+1, C, N), device=x_init.device).type(torch.float32)
        sol[:, 0:1] = x_init
        with torch.set_grad_enabled(enable_grad):
            for i in range(steps):
                cur_t = torch.tensor(i * step_size).to(x_init.device).type(torch.float32)
                sol_i = integ_func(sol[:, i:i+1], ode_func(cur_t, sol[:, i:i+1]), step_size)
                sol[:, i+1:i+2] = sol_i
        
        if sol.ndim == 4:
            sol = sol[None, ...]
        sol = sol.permute(0, 2, 1, 3, 4)
                    
        if return_intermediates:
            step_grid = torch.floor(time_grid / step_size).type(torch.int64)
            return sol[:, step_grid]
        else:
            return sol[:, -1:]
    
    
    def symplectic_euler_step(self, x, v, dt):
        x, v = x.permute(0, 1, 3, 2), v.permute(0, 1, 3, 2)
        size = x.shape
        x_t, v_t = torch.reshape(x, shape=(-1, size[-1])), torch.reshape(v, shape=(-1, size[-1]))
        x_t_plus = SO3ODESolver.SO3.compose(x_t.to('cpu'), (v_t * dt).to('cpu'))
        
        x_t_plus = torch.reshape(x_t_plus, shape=size).type(torch.float32).to(x_t.device)
        x_t_plus = x_t_plus.permute(0, 1, 3, 2)
        return x_t_plus
    
    def euler_step(self, x, v, dt):
        y = x + v * dt
        return y



class SO3ODESolverWithVel(SO3ODESolver):
    def sample(self, x_init: Tensor, last_vel: Tensor, step_size: Optional[float], method: str = "euler",
               atol: float = 1e-5, rtol: float = 1e-5, time_grid: Tensor = torch.tensor([0.0, 1.0]), 
               time_grid_scale: bool = True, return_intermediates: bool = False, enable_grad: bool = False,
               hz: int = 50, **model_extras) -> Union[Tensor, Sequence[Tensor]]:

        B, _ , C, N = x_init.shape
        
        def ode_func(t, x, lv):
            if not time_grid_scale:
                t = t / hz
            return self.velocity_model(x=x, t=t, last_vel=lv, **model_extras)
        
        if method == "symplectic_euler":
            integ_func = self.symplectic_euler_step
        elif method == "euler":
            integ_func = self.euler_step

        time_grid = time_grid.to(x_init.device)
        if time_grid_scale:
            time_grid = time_grid / hz
        steps = math.floor(time_grid[-1] / step_size)
        
        sol = torch.zeros(size=(B, steps+1, C, N), device=x_init.device).type(torch.float32)
        sol[:, 0:1], v_init = x_init, last_vel
        with torch.set_grad_enabled(enable_grad):
            for i in range(steps):
                cur_t = torch.tensor(i * step_size).to(x_init.device).type(torch.float32)
                v_init = ode_func(cur_t, sol[:, i:i+1], lv=v_init)
                sol_i = integ_func(sol[:, i:i+1], v_init, step_size)
                sol[:, i+1:i+2] = sol_i
        
        if sol.ndim == 4:
            sol = sol[None, ...]
        sol = sol.permute(0, 2, 1, 3, 4)
                 
        if return_intermediates:
            step_grid = torch.floor(time_grid / step_size).type(torch.int64)
            return sol[:, step_grid]
        else:
            return sol[:, -1:]
'''        



if __name__ == "__main__":    
    ode_solver = ODESolver(velocity_model=lambda x, t: 0.1*t)
    so3ode_solver = SO3ODESolver(velocity_model=lambda x, t: 0.1*t)
    
    time_grid = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    
    # x_init = torch.rand(size=(1, 1, 1, 1))
    # print(torch.squeeze(x_init))
    x_init = torch.ones(size=(1, 1, 1, 1)) * 0.7
    print(torch.squeeze(x_init))
    
    y1 = ode_solver.sample(time_grid=time_grid, time_grid_scale=1.0, x_init=x_init, \
                          method='euler', step_size=0.1, return_intermediates=True, hz=1.0)
    print(torch.squeeze(y1))
    print(y1.shape)
    
    y2 = so3ode_solver.sample(time_grid=time_grid, time_grid_scale=1.0, x_init=x_init, \
                             method='euler', step_size=0.1, return_intermediates=True, hz=1.0)
    print(torch.squeeze(y2))