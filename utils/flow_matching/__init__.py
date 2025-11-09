from .ode_solver import Solver, ODESolver, SO3ODESolver, SO3ODESolverWithVel
from .velocity_path import FlowPathBase, VelocityPath, LatentVelocityPath, MixVelocityPath

flow_path_dict = {
    "velocity_path": VelocityPath,
    "latent_velocity_path": LatentVelocityPath,
    "mix_velocity_path": MixVelocityPath
}