# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from brax.training import networks
from flax import linen


def make_actor_model(parametric_action_distribution, obs_size):
    return networks.make_model(
        [512, 256, 128, 64, parametric_action_distribution.param_size],
        obs_size,
        activation=linen.swish)


def make_critic_model(obs_size):
    return networks.make_model(
        [256, 128, 64, 32, 1],
        obs_size,
        activation=linen.swish)
