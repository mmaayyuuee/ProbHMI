from .graphflow import FlowBaseNet
from .graphflow import FlowNet, DctFlowNet
from .graphflow import PairFlowContainer
from .rnn_dynamics import RecurrentDynamics
from .rnn_dynamics import RecurrentDynamics_V1, RecurrentDynamics_V2, RecurrentDynamics_V3
from .rnn_dynamics import RecurrentDynamics_V4, RecurrentDynamics_V5
from .linear_dynamics import LinearDynamics
from .linear_dynamics import LinearDynamics_V1, LinearDynamics_V2
from .stgcn import STGCN, DctNet, DctSTGCN, FowardSTGCN
from .cfm_dynamics import FlowMatchingDynamicsBase, FlowMatchingDynamics_V1

flow_model_dict = {
    1: FlowNet,
    5: DctFlowNet
}

recurrent_dyna_dict = {
    1: RecurrentDynamics_V1,
    2: RecurrentDynamics_V2,
    3: RecurrentDynamics_V3,
    4: RecurrentDynamics_V4,
    5: RecurrentDynamics_V5
}

linear_dyna_dict = {
    1: LinearDynamics_V1,
    2: LinearDynamics_V2
}


flowmatching_dyna_dict = {
    1: FlowMatchingDynamics_V1
}
