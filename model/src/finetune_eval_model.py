import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import Concat as C
from mindspore.ops import functional as F
from mindspore import context
from .bert_model import BertModel,BertModelEval,BertModelAllSeqs
import mindspore.numpy as mnp
from mindspore.ops import MaskedSelect as Mask
from .bert_model import CreateAttentionMaskFromInputMask,BertSelfAttention
import mindspore.ops as ops
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, TruncatedNormal

class BertCLSModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLSModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        if not is_training:
            self.log_softmax = P.Softmax(axis=-1)
        else:
            self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1.0 - dropout_prob)
        self.assessment_method = assessment_method

    def construct(self, input_ids, input_mask, token_type_id):

        sequence_output, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        logits = self.log_softmax(logits)
        return logits

    def predict(self, input_ids, input_mask, token_type_id):
        sequence_output, pooled_output, _,all_sequence_output,all_polled_output = self.bert(input_ids, token_type_id, input_mask,return_all_encoders=True)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        logits = self.log_softmax(logits)
        return logits,sequence_output, pooled_output, all_sequence_output,all_polled_output

    def attention(self, input_ids, input_mask, token_type_id):
        attention=self.bert.get_attention(input_ids, token_type_id, input_mask)
        return attention

class BertRegModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=1, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertRegModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, 1, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.sigmoid = nn.Sigmoid()

    def construct(self, input_ids, input_mask, token_type_id):
        sequence_output, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        cls = self.dropout(pooled_output)
        logits=self.dense_1(cls)
        logits=self.sigmoid(logits)
        return logits

    def predict(self, input_ids, input_mask, token_type_id):
        sequence_output, pooled_output, _,all_sequence_output,all_polled_output = self.bert(input_ids, token_type_id, input_mask,return_all_encoders=True)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits=self.sigmoid(logits)
        return logits,sequence_output, pooled_output, all_sequence_output,all_polled_output

class GetMaskedLMOutput(nn.Cell):
    """
    Get masked lm output.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        Tensor, masked lm output.
    """

    def __init__(self, config,is_training):
        super(GetMaskedLMOutput, self).__init__()
        self.width = config.hidden_size
        self.reshape = P.Reshape()
        self.gather = P.Gather()

        weight_init = TruncatedNormal(config.initializer_range)
        self.dense = nn.Dense(self.width,
                              config.hidden_size,
                              weight_init=weight_init,
                              activation=config.hidden_act).to_float(config.compute_type)
        self.layernorm = nn.LayerNorm((config.hidden_size,)).to_float(config.compute_type)
        self.output_bias = Parameter(
            initializer(
                'zero',
                config.vocab_size))
        self.matmul = P.MatMul(transpose_b=True)
        if is_training:
            self.log_softmax = nn.LogSoftmax(axis=-1)
        else:
            self.log_softmax = nn.Softmax(axis=-1)
        self.shape_flat_offsets = (-1, 1)
        self.last_idx = (-1,)
        self.shape_flat_sequence_tensor = (-1, self.width)
        self.cast = P.Cast()
        self.compute_type = config.compute_type
        self.dtype = config.dtype

    def construct(self,
                  input_tensor,
                  output_weights,
                  positions):
        """Get output log_probs"""
        input_shape = P.Shape()(input_tensor)
        rng = F.tuple_to_array(F.make_range(input_shape[0]))
        flat_offsets = self.reshape(rng * input_shape[1], self.shape_flat_offsets)
        flat_position = self.reshape(positions + flat_offsets, self.last_idx)
        flat_sequence_tensor = self.reshape(input_tensor, self.shape_flat_sequence_tensor)
        input_tensor = self.gather(flat_sequence_tensor, flat_position, 0)
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)
        input_tensor = self.dense(input_tensor)
        input_tensor = self.layernorm(input_tensor)
        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)
        logits = logits + self.output_bias
        log_probs = self.log_softmax(logits)

        return log_probs

class BertMaskModel(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings,dropout_prob=0.0):
        super(BertMaskModel, self).__init__()
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cls1 = GetMaskedLMOutput(config,is_training)
        self.dropout = nn.Dropout(1 - dropout_prob)

    def construct(self, input_ids, input_mask, token_type_id,
                  masked_lm_positions):
        sequence_output, pooled_output, embedding_table = \
            self.bert(input_ids, token_type_id, input_mask)
        sequence_output = self.dropout(sequence_output)
        prediction_scores = self.cls1(sequence_output,
                                      embedding_table,
                                      masked_lm_positions)
        return prediction_scores

    def attention(self, input_ids, input_mask, token_type_id,
                  masked_lm_positions):
        attention=self.bert.get_attention(input_ids, token_type_id, input_mask)
        return attention

class BertCLSModelEval(nn.Cell):
    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLSModelEval, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
            self.log_softmax=nn.Softmax(axis=-1)
        else:
            self.log_softmax = P.LogSoftmax(axis=-1)
        self.bert = BertModelEval(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.assessment_method = assessment_method

    def construct(self, input_ids, input_mask, token_type_id):
        sequence_output, pooled_output, _,all_sequence_output,all_polled_output = self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        if self.assessment_method != "spearman_correlation":
            logits = self.log_softmax(logits)
        return logits,pooled_output,sequence_output,all_polled_output,all_sequence_output

class BertSEQModel(nn.Cell):
    def __init__(self, config, is_training, num_labels=11, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSEQModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
            self.log_softmax=nn.Softmax(axis=-1)
        else:
            self.log_softmax = P.LogSoftmax(axis=-1)
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        if with_lstm:
            self.lstm_hidden_size = config.hidden_size // 2
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.reshape = P.Reshape()
        self.shape = (-1, config.hidden_size)
        self.with_lstm = with_lstm
        self.origin_shape = (-1, config.seq_length, self.num_labels)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        seq = self.dropout(sequence_output)
        if self.with_lstm:
            batch_size = input_ids.shape[0]
            data_type = self.dtype
            hidden_size = self.lstm_hidden_size
            h0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            c0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            seq, _ = self.lstm(seq, (h0, c0))

        seq = self.reshape(seq, self.shape)
        logits = self.dense_1(seq)
        logits = self.cast(logits, self.dtype)
        return_value = self.log_softmax(logits)
        return return_value

class BertSEQModelEval(nn.Cell):
    """
    This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
    The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=11, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSEQModelEval, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
            self.log_softmax=nn.Softmax(axis=-1)
        else:
            self.log_softmax = P.LogSoftmax(axis=-1)
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        if with_lstm:
            self.lstm_hidden_size = config.hidden_size // 2
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.reshape = P.Reshape()
        self.shape = (-1, config.hidden_size)
        self.with_lstm = with_lstm
        self.origin_shape = (-1, config.seq_length, self.num_labels)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        seq = self.dropout(sequence_output)
        if self.with_lstm:
            batch_size = input_ids.shape[0]
            data_type = self.dtype
            hidden_size = self.lstm_hidden_size
            h0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            c0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            seq, _ = self.lstm(seq, (h0, c0))

        seq = self.reshape(seq, self.shape)
        logits = self.dense_1(seq)
        logits = self.cast(logits, self.dtype)
        return_value = self.log_softmax(logits)
        return return_value,sequence_output

