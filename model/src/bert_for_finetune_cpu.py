
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from .bert_for_training import clip_grad

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0


class BertFinetuneCellCPU(nn.TrainOneStepCell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(BertFinetuneCellCPU, self).__init__(network, optimizer, sens)

        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()

    def set_sens(self, value):
        self.sens = value

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids,
                  sens=None):
        """Bert Finetune"""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


class BertSquadCellCPU(nn.TrainOneStepCell):
    """
    Encapsulation class of bert squad network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(BertSquadCellCPU, self).__init__(network, optimizer, sens)

        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()

    def set_sens(self, value):
        self.sens = value

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  start_position,
                  end_position,
                  unique_id,
                  is_impossible,
                  sens=None):
        """BertSquad"""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            start_position,
                            end_position,
                            unique_id,
                            is_impossible)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 start_position,
                                                 end_position,
                                                 unique_id,
                                                 is_impossible,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss
