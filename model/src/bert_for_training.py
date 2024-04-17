# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Bert for pretraining."""
import numpy as np

import mindspore.nn as nn
from mindspore.common.initializer import initializer, TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
# from mindspore.common.api import jit
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore import context
from .bert_model import BertModel

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad



grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


cast = P.Cast()
add_grads = C.MultitypeFuncGraph("add_grads")


@add_grads.register("Tensor", "Tensor")
def _add_grads(accu_grad, grad):
    return accu_grad + cast(grad, mstype.float32)

update_accu_grads = C.MultitypeFuncGraph("update_accu_grads")

@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, cast(grad, mstype.float32)))

accumulate_accu_grads = C.MultitypeFuncGraph("accumulate_accu_grads")

@accumulate_accu_grads.register("Tensor", "Tensor")
def _accumulate_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign_add(accu_grad, cast(grad, mstype.float32)))


zeroslike = P.ZerosLike()
reset_accu_grads = C.MultitypeFuncGraph("reset_accu_grads")


@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, zeroslike(accu_grad)))


