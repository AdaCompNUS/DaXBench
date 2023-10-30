# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# import torch
# import torch.nn as nn
import jax.numpy as jnp


class AverageMeter():
    def __init__(self, in_shape, max_size):
        self.max_size = max_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        size = len(values.flatten())
        if size == 0:
            return
        new_mean = jnp.mean(values)
        size = jnp.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean
