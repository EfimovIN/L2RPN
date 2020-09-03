# import os
# from l2rpn_baselines.DuelQSimple import DuelQSimple, DuelQ_NNParam

import grid2op
from grid2op.Reward import L2RPNReward
from l2rpn_baselines.utils import TrainingParam 
from l2rpn_baselines.DuelQLeapNet import train, DuelQLeapNet
from l2rpn_baselines.DuelQLeapNet.LeapNet_NNParam import LeapNet_NNParam
import tensorflow as tf
from l2rpn_baselines.DuelQLeapNet.DuelQLeapNet_NN import DuelQLeapNet_NN
import os

name = "FirstDQLN"


def make_agent(env, submission_dir):
#     import pathlib
#     pathic = pathlib.Path().absolute()
#     submission_dir = f'{pathic}/submission/MODEL'
    
    load_path = submission_dir
    
    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    runner_params = env.get_params_for_runner()
#     runner_params["verbose"] = verbose

    if load_path is None:
        raise RuntimeError("Cannot evaluate a model if there is nothing to be loaded.")
    path_model, path_target_model = DuelQLeapNet_NN.get_path_model(load_path, name)
    nn_archi = LeapNet_NNParam.from_json(os.path.join(path_model, "nn_architecture.json"))

    # Run
    # Create agent
    agent = DuelQLeapNet(action_space=env.action_space,
                         name=name,
                         store_action=1,
                         nn_archi=nn_archi,
                         observation_space=env.observation_space)

    # Load weights from file
    agent.load(load_path)
    return agent
