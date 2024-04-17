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

'''
Bert finetune and evaluation script.
'''
import os
from sklearn import metrics
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import numpy as np
from src.bert_for_finetune import EarlyStoppingSaveBestMask
from src.bert_for_finetune_cpu import BertFinetuneCellCPU
from src.bert_for_finetune import BertFinetuneCellMask, BertMask
from src.dataset import create_mask_dataset
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, BertLearningRate
from src.model_utils.config import config as args_opt, optimizer_cfg, bert_net_cfg
from glob import glob
import pandas as pd
from src import tokenization
from tqdm import tqdm
import pickle
from src.script_utils import cal_matrix


_cur_dir = os.getcwd()




def do_train(network=None, load_checkpoint_path="", save_checkpoint_path=""):
    """ do train """
    if load_checkpoint_path==None or len(load_checkpoint_path)==0:
        print("load_checkpoint_path is None")
    else:
        param_dict = load_checkpoint(load_checkpoint_path)
        load_param_into_net(network, param_dict)

    epoch_num = args_opt.epoch_num

    train_data_file_path=os.path.join(args_opt.data_url,"train.mindrecord")
    ds_train = create_mask_dataset(batch_size=args_opt.train_batch_size,
                                       data_file_path=train_data_file_path,
                                       do_shuffle=(args_opt.train_data_shuffle.lower() == "true"))

    val_data_file_path=os.path.join(args_opt.data_url,"val.mindrecord")
    ds_val = create_mask_dataset(batch_size=args_opt.eval_batch_size,
                                           data_file_path=val_data_file_path,
                                           do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))

    steps_per_epoch = ds_train.get_dataset_size()
    # optimizer

    lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                   decay_steps=steps_per_epoch * epoch_num,
                                   power=optimizer_cfg.AdamWeightDecay.power)
    params = network.trainable_params()
    decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
    other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0}]
    optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)


    if ms.get_context("device_target") == "CPU":
        netwithgrads = BertFinetuneCellCPU(network, optimizer=optimizer)
    else:
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
        netwithgrads = BertFinetuneCellMask(network, optimizer=optimizer, scale_update_cell=update_cell)

    model = Model(netwithgrads)
    callbacks = [TimeMonitor(ds_train.get_dataset_size()), LossCallBack(ds_train.get_dataset_size())]

    callbacks.append(
        EarlyStoppingSaveBestMask(network, ds_val, early_stopping_rounds=args_opt.early_stopping_rounds, save_checkpoint_path=os.path.join(save_checkpoint_path,args_opt.task_name+"_Best_Model.ckpt")))

    model.train(epoch_num, ds_train, callbacks=callbacks)

def do_eval(dataset=None, network=None,  load_checkpoint_path=""):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_predict = network(bert_net_cfg, False)
    net_for_predict.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_predict, param_dict)
    columns_list = ["input_ids", "input_mask", "segment_ids","mask_lm_positions","mask_lm_ids","mask_lm_weights"]
    true_labels=[]
    pred_labels=[]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, mask_lm_positions, mask_lm_ids, mask_lm_weights = input_data
        logits= net_for_predict.predict(input_ids, input_mask, token_type_id, mask_lm_positions)
        pred_labels.append(logits.asnumpy())
        true_labels.append(mask_lm_ids.asnumpy()[0])
    pred_labels=np.vstack(pred_labels)
    true_labels=np.concatenate(true_labels)
    print_result=cal_matrix(true_labels,pred_labels,25,load_checkpoint_path)





import random
rng = random.Random(42)

def generate_predict_seq(data,tokenizer,seq_len):
    tokens_a = list(data)
    num_to_mask = int(len(tokens_a) * args_opt.mask_prob)
    mask_lm_positions = rng.sample(range(len(tokens_a)), num_to_mask)
    mask_lm_positions.sort()
    for i in mask_lm_positions:
        tokens_a[i] = '[MASK]'

    tokens_a = tokenizer.tokenize(tokens_a)

    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)
    tokens.extend(tokens_a)
    segment_ids.extend([0]*len(tokens_a))

    tokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenization.convert_tokens_to_ids(args_opt.vocab_file, tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    mask_lm_positions = [i+1 for i in mask_lm_positions]

    assert len(input_ids) == seq_len
    assert len(input_mask) == seq_len
    assert len(segment_ids) == seq_len

    return input_ids, input_mask, segment_ids, mask_lm_positions,tokens_a

def read_fasta( input_file, quotechar=None):
    seq_id=""
    seq=""
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                assert len(seq_id)==len(seq)==0
                seq_id=line.strip()
            else:
                seq+=line.strip()

    return seq_id,seq

def do_predict(seq_len=1024, network=None,  load_checkpoint_path="",tokenizer=None):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_predict = network(bert_net_cfg, False)
    net_for_predict.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_predict, param_dict)

    seq_id,predict_seq = read_fasta(args_opt.data_url)
    data_file_name=args_opt.data_url.split("/")[-1].strip(".fasta")

    write_data=[]
    word_list=list(tokenizer.vocab_dict.keys())
    fasta_result=[]

    fasta_result.append(">" + seq_id + "_wt"  + "\n" + predict_seq)

    for mask_iter in tqdm(range(int(args_opt.predict_mask_num))):
        input_ids, input_mask, token_type_id, mask_lm_positions,truncate_token_a=generate_predict_seq(predict_seq,tokenizer,seq_len)
        logits = net_for_predict.predict(ms.Tensor([input_ids]), ms.Tensor([input_mask]), ms.Tensor([token_type_id]), ms.Tensor([mask_lm_positions])).asnumpy()
        real_seq=list(predict_seq)
        pred_seq=list(predict_seq)
        for i in range(len(mask_lm_positions)):
            pred_seq[mask_lm_positions[i]-1]=word_list[np.argmax(logits[i])]
        write_dict={
            "seq_id":seq_id+"_"+str(mask_iter),
            "seq":real_seq,
            "pred_seq":pred_seq,
            "truncate_token":truncate_token_a,
            "mask_lm_positions":mask_lm_positions,
            "logits":logits
        }
        fasta_result.append(">"+seq_id+"_"+str(mask_iter)+"\n"+"".join(pred_seq))

        write_data.append(write_dict)
    pickle.dump(write_data,open(args_opt.output_url+"/"+data_file_name+".pkl","wb"))
    with open(args_opt.output_url+"/"+data_file_name+".fasta","w") as f:
        f.write("\n".join(fasta_result))

def do_attention(seq_len=1024, network=None,  load_checkpoint_path="",tokenizer=None):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_predict = network(bert_net_cfg, False)

    params = net_for_predict.get_parameters()
    total_params = 0
    for p in params:
        total_params += p.size
    print(f"Total parameters: {total_params}")

    net_for_predict.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_predict, param_dict)

    seq_id,predict_seq = read_fasta(args_opt.data_url)
    data_file_name=args_opt.data_url.split("/")[-1].strip(".fasta")

    print("This script do not use nohup")
    mut_seq_id=input("Please input the seq_id, split with ',' :")
    mut_seq_id=mut_seq_id.split(",")

    write_data=[]
    word_list=list(tokenizer.vocab_dict.keys())

    for mask_iter in tqdm(range(int(args_opt.predict_mask_num))):
        input_ids, input_mask, token_type_id, mask_lm_positions,truncate_token_a=generate_predict_seq(predict_seq,tokenizer,seq_len)
        if str(mask_iter) not in mut_seq_id:
            continue
        else:
            print("generate ",mask_iter,flush=True)
        logits = net_for_predict.predict(ms.Tensor([input_ids]), ms.Tensor([input_mask]), ms.Tensor([token_type_id]), ms.Tensor([mask_lm_positions])).asnumpy()
        attention = net_for_predict.attention(ms.Tensor([input_ids]), ms.Tensor([input_mask]), ms.Tensor([token_type_id]), ms.Tensor([mask_lm_positions]))
        attention=[a.asnumpy() for a in attention]
        real_seq=list(predict_seq)
        pred_seq=list(predict_seq)
        for i in range(len(mask_lm_positions)):
            pred_seq[mask_lm_positions[i]-1]=word_list[np.argmax(logits[i])]
        write_dict={
            "seq_id":seq_id+"_"+str(mask_iter),
            "seq":"".join(real_seq),
            "pred_seq":"".join(pred_seq),
            "truncate_token":truncate_token_a,
            "mask_lm_positions":mask_lm_positions,
            "logits":logits,
            "attention":attention
        }
        print("pred_seq:",write_dict["pred_seq"],flush=True)
        print("attention:",np.array(write_dict["attention"]).shape,flush=True)

        write_data.append(write_dict)
    pickle.dump(write_data,open(args_opt.output_url+"/"+data_file_name+"_attention.pkl","wb"))

def run_mask():
    """run classifier task"""
    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        context.set_context(enable_graph_kernel=True)
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    elif target == "CPU":
        if args_opt.use_pynative_mode:
            context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=args_opt.device_id)
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU", device_id=args_opt.device_id)
    else:
        raise Exception("Target error, CPU or GPU or Ascend is supported.")

    val_data_file_path=os.path.join(args_opt.data_url,"val.mindrecord")
    test_data_file_path=os.path.join(args_opt.data_url,"test.mindrecord")

    if args_opt.do_train==True:
        netwithloss = BertMask(bert_net_cfg, True, dropout_prob=0.1)

        print("==============================================================")
        print("processor_name: {}".format(args_opt.device_target))
        print("test_name: {}".format(args_opt.task_name))
        print("batch_size: {}".format(args_opt.train_batch_size))

        do_train( netwithloss, args_opt.load_checkpoint_url, args_opt.output_url)

    if args_opt.do_eval == True:
        if args_opt.do_train == True:
            finetune_ckpt_url = args_opt.output_url
        else:
            finetune_ckpt_url = args_opt.load_checkpoint_url
        if finetune_ckpt_url.endswith(".ckpt"):
            best_ckpt = finetune_ckpt_url
        else:
            load_finetune_checkpoint_dir = make_directory(finetune_ckpt_url)
            best_ckpt = LoadNewestCkpt(load_finetune_checkpoint_dir, args_opt.task_name)


        ds_val = create_mask_dataset(batch_size=args_opt.eval_batch_size,
                                               data_file_path=val_data_file_path,
                                               do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))

        ds_test = create_mask_dataset(batch_size=args_opt.eval_batch_size,
                                           data_file_path=test_data_file_path,
                                           do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))

        print("======Val======")
        if os.path.exists(val_data_file_path) == True:
            do_eval(ds_val, BertMask,  load_checkpoint_path=best_ckpt)
        print("======Test======")
        do_eval(ds_test, BertMask, load_checkpoint_path=best_ckpt)

    if args_opt.do_predict==True:
        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file, do_lower_case=False)
        finetune_ckpt_url = args_opt.load_checkpoint_url
        if args_opt.do_eval==False:
            if finetune_ckpt_url.endswith(".ckpt") ==False:
                raise "For predict, if do_eval==False, you should select only one checkpoint file and this file should end with .ckpt"
            else:
                best_ckpt=finetune_ckpt_url
        do_predict(bert_net_cfg.seq_length, BertMask,  load_checkpoint_path=best_ckpt,
                   tokenizer=tokenizer)

    if args_opt.do_attention==True:
        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file, do_lower_case=False)
        finetune_ckpt_url = args_opt.load_checkpoint_url
        if args_opt.do_eval==False:
            if finetune_ckpt_url.endswith(".ckpt") ==False:
                raise "For predict, if do_eval==False, you should select only one checkpoint file and this file should end with .ckpt"
            else:
                best_ckpt=finetune_ckpt_url

        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file, do_lower_case=False)
        do_attention(bert_net_cfg.seq_length, BertMask,  load_checkpoint_path=best_ckpt,
                   tokenizer=tokenizer)

    print("FINISH !!!")


if __name__ == "__main__":
    print(args_opt)
    bert_net_cfg.num_hidden_layers=bert_net_cfg.num_hidden_layers-int(args_opt.cut_layer_num)
    print(bert_net_cfg)
    run_mask()
