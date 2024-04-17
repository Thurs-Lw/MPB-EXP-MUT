import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from .bert_for_training import clip_grad
from .finetune_eval_model import BertCLSModel, BertSEQModel, BertSEQModelEval,BertCLSModelEval,BertRegModel,BertMaskModel
from .utils import CrossEntropyCalculation,FocalLossCalculation,MaskLoss
import numpy as np
from mindspore import save_checkpoint
import random
from sklearn import metrics
from scipy import stats
import time


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
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


class BertFinetuneCell(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertFinetuneCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()

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
        if sens is None:
            scaling_sens = self.scale_sense
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)

class BertFinetuneCellMask(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertFinetuneCellMask, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  mask_lm_positions,
                  mask_lm_ids,
                  mask_lm_weights,
                  sens=None):
        """Bert Finetune"""

        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            mask_lm_positions,
                            mask_lm_ids,
                            mask_lm_weights)
        if sens is None:
            scaling_sens = self.scale_sense
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                    mask_lm_positions,
                                                    mask_lm_ids,
                                                    mask_lm_weights,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)


class BertCLS(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method="", loss_func="CrossEntropy",label_percent=None,loss_gama=2.0):
        super(BertCLS, self).__init__()
        self.bert = BertCLSModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings,
                                 assessment_method)
        if loss_func=="CrossEntropy":
            self.loss = CrossEntropyCalculation(is_training)
        elif loss_func=="Focal":
            self.loss = FocalLossCalculation(weight=Tensor(label_percent), gamma=loss_gama, reduction='mean',is_training=is_training)
        else:
            raise "Error Loss Name"
        self.num_labels = num_labels
        self.assessment_method = assessment_method
        self.is_training = is_training

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.is_training:
            return_value = self.loss(logits, label_ids, self.num_labels)
        else:
            return_value = logits
        return return_value

    def predict(self, input_ids, input_mask, token_type_id):
        self.set_train(False)  # Set the model to evaluation mode
        logits = self.bert(input_ids, input_mask, token_type_id)
        return logits

    def feature(self, input_ids, input_mask, token_type_id):
        self.set_train(False)  # Set the model to evaluation mode
        logits,sequence_output, pooled_output, all_sequence_output,all_polled_output = self.bert.predict(input_ids, input_mask, token_type_id)
        return logits,sequence_output, pooled_output, all_sequence_output,all_polled_output

    def attention(self, input_ids, input_mask, token_type_id):
        self.set_train(False)  # Set the model to evaluation mode
        attention = self.bert.attention(input_ids, input_mask, token_type_id)
        return attention

class BertReg(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, num_labels=1, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertReg, self).__init__()
        self.bert = BertRegModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings,
                                 assessment_method)
        self.is_training = is_training
        self.mse=nn.MSELoss()

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.is_training:
            loss = self.mse(logits, label_ids)
        else:
            loss = logits * 1.0
        return loss

    def predict(self, input_ids, input_mask, token_type_id):
        self.set_train(False)  # Set the model to evaluation mode
        logits = self.bert(input_ids, input_mask, token_type_id)
        return logits

    def feature(self, input_ids, input_mask, token_type_id):
        self.set_train(False)  # Set the model to evaluation mode
        logits,sequence_output, pooled_output, all_sequence_output,all_polled_output = self.bert.predict(input_ids, input_mask, token_type_id)
        return logits,sequence_output, pooled_output, all_sequence_output,all_polled_output

class BertCLSEval(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLSEval, self).__init__()
        self.bert = BertCLSModelEval(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings,
                                 assessment_method)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.assessment_method = assessment_method
        self.is_training = is_training

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits,pooled_output,sequence_output,all_polled_output,all_sequence_output = self.bert(input_ids, input_mask, token_type_id)
        if self.assessment_method == "spearman_correlation":
            if self.is_training:
                loss = self.loss(logits, label_ids)
            else:
                loss = logits
        else:
            loss = self.loss(logits, label_ids, self.num_labels)
        return loss,pooled_output,sequence_output,all_polled_output,all_sequence_output

class BertSeq(nn.Cell):
    """
    Train interface for sequence labeling finetuning task.
    """

    def __init__(self, config,  is_training, num_labels=11, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False,loss_func="CrossEntropy",label_percent=None,loss_gama=2.0):
        super(BertSeq, self).__init__()
        self.bert = BertSEQModel(config, is_training, num_labels,  with_lstm, dropout_prob,
                                 use_one_hot_embeddings)
        if loss_func=="CrossEntropy":
            self.loss = CrossEntropyCalculation(is_training)
        elif loss_func=="Focal":
            self.loss = FocalLossCalculation(weight=Tensor(label_percent), gamma=loss_gama, reduction='mean',is_training=is_training)
        else:
            raise "Error Loss Name"
        self.loss_name=loss_func
        self.num_labels = num_labels

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        loss = self.loss(logits, label_ids, self.num_labels)

        return loss

class BertMask(nn.Cell):
    """
    Train interface for sequence labeling finetuning task.
    """

    def __init__(self, config,  is_training,
                 dropout_prob=0.0, use_one_hot_embeddings=False,loss_func="CrossEntropy",label_percent=None,loss_gama=2.0):
        super(BertMask, self).__init__()
        self.bert = BertMaskModel(config, is_training, use_one_hot_embeddings,dropout_prob=dropout_prob)
        self.loss = MaskLoss(config,is_training)
        self.loss_name=loss_func

    def construct(self, input_ids, input_mask, token_type_id, mask_lm_positions,mask_lm_ids,mask_lm_weights):
        logits = self.bert(input_ids, input_mask, token_type_id,mask_lm_positions)
        loss = self.loss(logits,mask_lm_ids,mask_lm_weights)
        return loss

    def predict(self, input_ids, input_mask, token_type_id, mask_lm_positions):
        self.set_train(False)  # Set the model to evaluation mode
        logits = self.bert(input_ids, input_mask, token_type_id,mask_lm_positions)
        return logits

    def attention(self, input_ids, input_mask, token_type_id, mask_lm_positions):
        self.set_train(False)  # Set the model to evaluation mode
        attention = self.bert.attention(input_ids, input_mask, token_type_id,mask_lm_positions)
        return attention

# class BertMask(nn.Cell):
#     """
#     Train interface for sequence labeling finetuning task.
#     """
#
#     def __init__(self, config,  is_training,
#                  dropout_prob=0.0, use_one_hot_embeddings=False,loss_func="CrossEntropy",label_percent=None,loss_gama=2.0):
#         super(BertMask, self).__init__()
#         self.bert = BertSEQModel(config, is_training, use_one_hot_embeddings=use_one_hot_embeddings,dropout_prob=dropout_prob,num_labels=25)
#         self.loss = CrossEntropyCalculation(is_training)
#         self.loss_name=loss_func
#
#     def construct(self, input_ids, input_mask, token_type_id, mask_lm_positions,mask_lm_ids,mask_lm_weights):
#         logits = self.bert(input_ids, input_mask, token_type_id)
#         loss = self.loss(logits,input_ids,25)
#         return loss
#
#     def predict(self, input_ids, input_mask, token_type_id, mask_lm_positions):
#         self.set_train(False)  # Set the model to evaluation mode
#         logits = self.bert(input_ids, input_mask, token_type_id)
#         return logits



class BertSeqEval(nn.Cell):
    """
    Train interface for sequence labeling finetuning task.
    """

    def __init__(self, config,  is_training, num_labels=11, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False,loss_func="CrossEntropy",label_percent=None,loss_gama=2.0):
        super(BertSeqEval, self).__init__()
        self.bert = BertSEQModelEval(config, is_training, num_labels,  with_lstm, dropout_prob,
                                 use_one_hot_embeddings)
        if loss_func=="CrossEntropy":
            self.loss = CrossEntropyCalculation(is_training)
        elif loss_func=="Focal":
            self.loss = FocalLossCalculation(weight=Tensor(label_percent), gamma=loss_gama, reduction='mean',is_training=is_training)
        else:
            raise "Error Loss Name"
        self.loss_name=loss_func
        self.num_labels = num_labels

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits,sequence_output = self.bert(input_ids, input_mask, token_type_id)
        loss = self.loss(logits, label_ids, self.num_labels)

        return loss,sequence_output

from mindspore.train.callback import Callback
import os

class EarlyStoppingSaveBest(Callback):
    def __init__(self, model, ds_val, early_stopping_rounds, save_checkpoint_path,num_labels=2):
        super(EarlyStoppingSaveBest, self).__init__()
        self.model = model
        self.ds_val = ds_val
        self.best_acc = 0.0
        self.best_epoch = 0
        self.early_stopping_rounds = early_stopping_rounds
        self.rounds_no_improve = 0
        self.save_checkpoint_path = save_checkpoint_path
        self.cur_epoch_num = 0
        self.num_labels=num_labels

    def epoch_end(self, run_context):
        self.model.set_train(False)  # Set the model to evaluation mode
        pred_labels = []
        true_labels = []
        Loss = CrossEntropyCalculation()
        total_loss=0
        for data in self.ds_val.create_dict_iterator():
            input_ids = data["input_ids"]
            input_mask = data["input_mask"]
            token_type_id = data["segment_ids"]
            label_ids = data["label_ids"]
            logits = self.model.predict(input_ids, input_mask, token_type_id)
            total_loss+=Loss(logits,label_ids,self.num_labels)
            logits=logits[0]
            labels=label_ids.asnumpy()[0][0]
            pred_labels.append(logits.asnumpy())
            true_labels.append(labels)

        pred_labels=Tensor(np.array(pred_labels),dtype=mstype.float32)
        true_labels=Tensor(np.array(true_labels),dtype=mstype.float32)
        # loss=total_loss.asnumpy()/len(true_labels)
        metric = nn.Accuracy('classification')
        metric.clear()
        metric.update(pred_labels, true_labels)
        acc = metric.eval()

        print(f"Validation accuracy: {acc}")

        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = self.cur_epoch_num
            self.rounds_no_improve = 0
            save_checkpoint(self.model,self.save_checkpoint_path)
            print(f"New best accuracy: {self.best_acc}, saved best model to :")
            print(self.save_checkpoint_path)
        else:
            self.rounds_no_improve += 1
            print(f"No improvement in accuracy for {self.rounds_no_improve} epochs (best acc = {self.best_acc}).")
        if self.rounds_no_improve >= self.early_stopping_rounds:
            print(f"Early stopping due to no improvement.")
            run_context.request_stop()

        self.model.set_train(True)
        self.cur_epoch_num += 1

class EarlyStoppingSaveBestMask(Callback):
    def __init__(self, model, ds_val, early_stopping_rounds, save_checkpoint_path):
        super(EarlyStoppingSaveBestMask, self).__init__()
        self.model = model
        self.ds_val = ds_val
        self.best_acc = 0.0
        self.best_epoch = 0
        self.early_stopping_rounds = early_stopping_rounds
        self.rounds_no_improve = 0
        self.save_checkpoint_path = save_checkpoint_path
        self.cur_epoch_num = 0

    def epoch_end(self, run_context):
        self.model.set_train(False)  # Set the model to evaluation mode
        pred_labels = []
        true_labels = []
        for data in self.ds_val.create_dict_iterator():
            input_ids = data["input_ids"]
            input_mask = data["input_mask"]
            token_type_id = data["segment_ids"]
            mask_lm_positions=data["mask_lm_positions"]
            mask_lm_ids=data["mask_lm_ids"]
            logits = self.model.predict(input_ids, input_mask, token_type_id,mask_lm_positions)

            pred_labels.append(logits.asnumpy())
            true_labels.append(mask_lm_ids.asnumpy()[0])
        pred_labels = np.vstack(pred_labels)
        true_labels = np.concatenate(true_labels)


        pred_labels=Tensor(pred_labels,dtype=mstype.float32)
        true_labels=Tensor(true_labels,dtype=mstype.float32)
        # loss=total_loss.asnumpy()/len(true_labels)
        metric = nn.Accuracy('classification')
        metric.clear()
        metric.update(pred_labels, true_labels)
        acc = metric.eval()

        print(f"Validation accuracy: {acc}")

        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = self.cur_epoch_num
            self.rounds_no_improve = 0
            save_checkpoint(self.model,self.save_checkpoint_path)
            print(f"New best accuracy: {self.best_acc}, saved best model to :")
            print(self.save_checkpoint_path)
        else:
            self.rounds_no_improve += 1
            print(f"No improvement in accuracy for {self.rounds_no_improve} epochs (best acc = {self.best_acc}).")
        if self.rounds_no_improve >= self.early_stopping_rounds:
            print(f"Early stopping due to no improvement.")
            run_context.request_stop()

        self.model.set_train(True)
        self.cur_epoch_num += 1


# class EarlyStoppingSaveBestMask(Callback):
#     def __init__(self, model, ds_val, early_stopping_rounds, save_checkpoint_path):
#         super(EarlyStoppingSaveBestMask, self).__init__()
#         self.model = model
#         self.ds_val = ds_val
#         self.best_acc = 0.0
#         self.best_epoch = 0
#         self.early_stopping_rounds = early_stopping_rounds
#         self.rounds_no_improve = 0
#         self.save_checkpoint_path = save_checkpoint_path
#         self.cur_epoch_num = 0
#         self.reshape = P.Reshape()
#         self.last_idx = (-1,)
#
#     def epoch_end(self, run_context):
#         self.model.set_train(False)  # Set the model to evaluation mode
#         pred_labels = []
#         true_labels = []
#         for data in self.ds_val.create_dict_iterator():
#             input_ids = data["input_ids"]
#             input_mask = data["input_mask"]
#             token_type_id = data["segment_ids"]
#             mask_lm_positions=data["mask_lm_positions"]
#             mask_lm_ids=data["mask_lm_ids"]
#             logits = self.model.predict(input_ids, input_mask, token_type_id,mask_lm_positions)
#             logits = logits.asnumpy()
#             labels = input_ids.asnumpy()[0]
#             pred_labels = np.vstack(pred_labels)
#             true_labels = np.array(true_labels)
#
#             pred_labels.append(logits)
#             true_labels.append(labels)
#
#         pred_labels = Tensor(np.array(pred_labels), dtype=mstype.float32)
#         true_labels = Tensor(np.array(true_labels), dtype=mstype.float32)
#         true_labels = self.reshape(true_labels, self.last_idx)
#
#         if self.cur_epoch_num == 0:
#             print(true_labels.shape)
#             print(pred_labels.shape)
#
#         # loss=total_loss.asnumpy()/len(true_labels)
#         metric = nn.Accuracy('classification')
#         metric.clear()
#         metric.update(pred_labels, true_labels)
#         acc = metric.eval()
#
#         print(f"Validation accuracy: {acc}")
#
#         if acc > self.best_acc:
#             self.best_acc = acc
#             self.best_epoch = self.cur_epoch_num
#             self.rounds_no_improve = 0
#             save_checkpoint(self.model,self.save_checkpoint_path)
#             print(f"New best accuracy: {self.best_acc}, saved best model to :")
#             print(self.save_checkpoint_path)
#         else:
#             self.rounds_no_improve += 1
#             print(f"No improvement in accuracy for {self.rounds_no_improve} epochs (best acc = {self.best_acc}).")
#         if self.rounds_no_improve >= self.early_stopping_rounds:
#             print(f"Early stopping due to no improvement.")
#             run_context.request_stop()
#
#         self.model.set_train(True)
#         self.cur_epoch_num += 1


class EarlyStoppingSaveBestRegress(Callback):
    def __init__(self, model, ds_val, early_stopping_rounds, save_checkpoint_path,num_labels=2):
        super(EarlyStoppingSaveBestRegress, self).__init__()
        self.model = model
        self.ds_val = ds_val
        self.best_mse = 9999999.999
        self.best_epoch = 0
        self.early_stopping_rounds = early_stopping_rounds
        self.rounds_no_improve = 0
        self.save_checkpoint_path = save_checkpoint_path
        self.cur_epoch_num = 0
        self.num_labels=num_labels
        self.start_time=time.time()
        self.cur_time=time.time()

    def epoch_end(self, run_context):
        self.model.set_train(False)  # Set the model to evaluation mode
        pred_labels = []
        true_labels = []

        pred=[]
        real=[]

        Loss = CrossEntropyCalculation()
        total_loss=0

        random_seed=random.randint(0,100)
        print_index=0

        for data in self.ds_val.create_dict_iterator():
            input_ids = data["input_ids"]
            input_mask = data["input_mask"]
            token_type_id = data["segment_ids"]
            label_ids = data["label_ids"]

            logits = self.model.predict(input_ids, input_mask, token_type_id)
            total_loss+=Loss(logits,label_ids,self.num_labels)
            logits=logits[0]
            labels=label_ids.asnumpy()[0][0]
            pred_labels.append(logits.asnumpy())
            true_labels.append(labels)

            if random_seed==print_index:
                print("Input Example(",random_seed,"):", input_ids.asnumpy()[0])
                print("Label Example(",random_seed,"):", label_ids.asnumpy()[0])
                print("Prediction Example(",random_seed,"):", logits.asnumpy())

            pred.append(logits.asnumpy()[0])
            real.append(label_ids.asnumpy()[0][0])

            print_index+=1

        pred_labels=Tensor(np.array(pred_labels),dtype=mstype.float32)
        true_labels=Tensor(np.array(true_labels),dtype=mstype.float32)

        # loss=total_loss.asnumpy()/len(true_labels)
        metric = nn.MSE()
        metric.clear()
        metric.update(pred_labels, true_labels)
        mse = metric.eval()

        r_square=metrics.r2_score(real,pred)
        pearson=stats.pearsonr(real,pred)[0]
        spearman=stats.spearmanr(real,pred)[0]

        print(f"Validation MSE: ",mse," R2: ",r_square," Pearson: ",pearson," Spearman: ",spearman)

        if mse < self.best_mse:
            self.best_mse = mse
            self.best_epoch = self.cur_epoch_num
            self.rounds_no_improve = 0
            save_checkpoint(self.model,self.save_checkpoint_path)
            print(f"New best MSE: {self.best_mse}, saved best model to :")
            print(self.save_checkpoint_path)
        else:
            self.rounds_no_improve += 1
            print(f"No improvement in MSE for {self.rounds_no_improve} epochs (best MSE = {self.best_mse}).")
        if self.rounds_no_improve >= self.early_stopping_rounds:
            print(f"Early stopping due to no improvement.")
            run_context.request_stop()

        self.model.set_train(True)
        self.cur_epoch_num += 1
        print("Epoch Time: ",time.time()-self.cur_time,"s, Total Time: ",time.time()-self.start_time,"s\n")
        self.cur_time=time.time()