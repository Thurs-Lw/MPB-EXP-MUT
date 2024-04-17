from sklearn import metrics
import os
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay
from mindspore.train.model import Model
from mindspore.train.callback import  TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.bert_for_finetune_cpu import BertFinetuneCellCPU
from src.bert_for_finetune import BertFinetuneCell, BertCLS,EarlyStoppingSaveBest
from src.dataset import create_classification_dataset
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, BertLearningRate
from src.model_utils.config import config as args_opt, optimizer_cfg, bert_net_cfg
import numpy as np
import pandas as pd
from src import tokenization
from src.script_utils import cal_matrix
from src.finetune_data_process import generate_predict_seq_1x
from tqdm import tqdm

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
    ds_train = create_classification_dataset(batch_size=args_opt.train_batch_size,
                                       data_file_path=train_data_file_path,
                                       do_shuffle=(args_opt.train_data_shuffle.lower() == "true"))

    val_data_file_path=os.path.join(args_opt.data_url,"val.mindrecord")
    ds_val = create_classification_dataset(batch_size=args_opt.eval_batch_size,
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
        netwithgrads = BertFinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)

    model = Model(netwithgrads)
    callbacks = [TimeMonitor(ds_train.get_dataset_size()), LossCallBack(ds_train.get_dataset_size())]

    callbacks.append(
        EarlyStoppingSaveBest(network, ds_val, early_stopping_rounds=args_opt.early_stopping_rounds, save_checkpoint_path=os.path.join(save_checkpoint_path,args_opt.task_name+"_Best_Model.ckpt"),num_labels=args_opt.num_class))

    model.train(epoch_num, ds_train, callbacks=callbacks)


def do_eval(dataset=None, network=None, num_class=None, load_checkpoint_path="",do_train=False):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(bert_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)
    softmax = ms.nn.Softmax()
    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    true_labels=[]
    pred_labels=[]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        logits= model.predict(input_ids, input_mask, token_type_id, label_ids)
        true_labels.append(label_ids.asnumpy()[0][0])
        logits=softmax(logits[0])
        pred_labels.append(logits.asnumpy())
    print_result=cal_matrix(true_labels,pred_labels,num_class,load_checkpoint_path)
    if do_train==True:
        return print_result["ACC"]

def do_predict(seq_len=1024, network=None, num_class=2, load_checkpoint_path="",tokenizer=None):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_predict = network(bert_net_cfg, False, num_class)
    net_for_predict.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_predict, param_dict)

    if args_opt.data_url.endswith(".csv"):
        print("data_url: ", args_opt.data_url, "is csv file")
        ds_predict = pd.read_csv(args_opt.data_url).to_dict("record")  # predict csv must have["id","seq"] or ["id_0","seq_0","id_1","seq_1"]
        data_file_name=args_opt.data_url.split("/")[-1].strip(".csv")
        if "seq" not in ds_predict[0].keys():
            raise "predict csv format ERROR"

    elif args_opt.data_url.endswith(".fasta") or args_opt.data_url.endswith(".fa") or args_opt.data_url.endswith(".fna"):
        print("data_url: ", args_opt.data_url, "is fasta file")
        ds_predict = []
        with open(args_opt.data_url, "r") as f:
            for l in f.readlines():
                if l.startswith(">"):
                    ds_predict.append({"id": l.strip().strip(">"), "seq": "","label":-1})
                else:
                    ds_predict[-1]["seq"] += l.strip()
            data_file_name = args_opt.data_url.split("/")[-1].replace(".fasta","").replace(".fa","").replace(".fna","")

    else:
        raise "predict data format ERROR"

    assert len(ds_predict)>0,"predict data get ERROR: None"


    if args_opt.return_sequence==True or args_opt.return_csv==True:
        write_data=[]
    if args_opt.print_predict==True:
        true_labels=[]
        pred_labels=[]
    for data in tqdm(ds_predict):
        input_ids, input_mask, token_type_id, label_ids,id,truncate_token_a=generate_predict_seq_1x(data,tokenizer,seq_len,args_opt)
        logits,sequence_output, pooled_output, all_sequence_output,all_polled_output = net_for_predict.feature(input_ids, input_mask, token_type_id)

        logits = logits[0]
        data["pred_label"] = np.argmax(logits.asnumpy())
        if args_opt.print_predict == True:
            true_labels.append(data["label"])
            pred_labels.append(logits.asnumpy())
        if args_opt.return_sequence == True:
            data["truncate_0"] = truncate_token_a
            data["feature"] = [all_sequence_output[0].asnumpy(), all_sequence_output[-1].asnumpy()]
        else:
            data["feature"] = None
        if args_opt.return_sequence == True or args_opt.return_csv == True:
            data["dense"] = logits.asnumpy()[1]
            write_data.append(data)
    if args_opt.print_predict == True:
        cal_matrix(true_labels, pred_labels, num_class, load_checkpoint_path, data_file_name)
    if args_opt.return_sequence == True:
        np.save(os.path.join(args_opt.output_url, data_file_name + "_predict_result.npy"), np.array(write_data))
    if args_opt.return_csv == True:
        pd.DataFrame(write_data).drop(["feature"], axis=1).to_csv(
            os.path.join(args_opt.output_url, data_file_name + "_predict_result.csv"))


def do_attention(seq_len=1024, network=None, num_class=2, load_checkpoint_path="",tokenizer=None):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_predict = network(bert_net_cfg, False, num_class)
    net_for_predict.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_predict, param_dict)

    if args_opt.data_url.endswith(".csv"):
        print("data_url: ", args_opt.data_url, "is csv file")
        ds_predict = pd.read_csv(args_opt.data_url).to_dict("record")  # predict csv must have["id","seq"] or ["id_0","seq_0","id_1","seq_1"]
        data_file_name=args_opt.data_url.split("/")[-1].strip(".csv")
        if "seq" not in ds_predict[0].keys():
            raise "predict csv format ERROR"

    elif args_opt.data_url.endswith(".fasta") or args_opt.data_url.endswith(".fa") or args_opt.data_url.endswith(".fna"):
        print("data_url: ", args_opt.data_url, "is fasta file")
        ds_predict = []
        with open(args_opt.data_url, "r") as f:
            for l in f.readlines():
                if l.startswith(">"):
                    ds_predict.append({"id": l.strip().strip(">"), "seq": "","label":-1})
                else:
                    ds_predict[-1]["seq"] += l.strip()
            data_file_name = args_opt.data_url.split("/")[-1].replace(".fasta","").replace(".fa","").replace(".fna","")
    else:
        raise "predict data format ERROR"
    assert len(ds_predict)>0,"predict data get ERROR: None"
    write_data=[]
    for data in tqdm(ds_predict):
        input_ids, input_mask, token_type_id, label_ids,id,truncate_token_a=generate_predict_seq_1x(data,tokenizer,seq_len,args_opt)
        logits,sequence_output, pooled_output, all_sequence_output,all_polled_output = net_for_predict.feature(input_ids, input_mask, token_type_id)
        attention = net_for_predict.attention(input_ids, input_mask, token_type_id)
        attention=[a.asnumpy() for a in attention]
        logits = logits[0]
        data["pred_label"] = np.argmax(logits.asnumpy())
        data["truncate_0"] = truncate_token_a
        data["feature"] = [all_sequence_output[0].asnumpy(), all_sequence_output[-1].asnumpy()]
        data["dense"] = logits.asnumpy()[1]
        data["attention"] = attention
        write_data.append(data)

    if "fold_" in load_checkpoint_path:
        fold_no=load_checkpoint_path.split("fold_")[-1].split("/")[0]
    else:
        fold_no="None"
    print("write_data: ",os.path.join(args_opt.output_url, data_file_name + "_predict_attention_fold_"+fold_no+".npy"))
    np.save(os.path.join(args_opt.output_url, data_file_name + "_cls_predict_attention.npy_fold_"+fold_no+".npy"), np.array(write_data))



def run_classifier():
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
        netwithloss = BertCLS(bert_net_cfg, True, num_labels=args_opt.num_class, dropout_prob=0.1)

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


        ds_val = create_classification_dataset(batch_size=args_opt.eval_batch_size,
                                               data_file_path=val_data_file_path,
                                               do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))

        ds_test = create_classification_dataset(batch_size=args_opt.eval_batch_size,
                                           data_file_path=test_data_file_path,
                                           do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))

        print("======Val======")
        if os.path.exists(val_data_file_path) == True:
            do_eval(ds_val, BertCLS, args_opt.num_class, load_checkpoint_path=best_ckpt)
        print("======Test======")
        do_eval(ds_test, BertCLS, args_opt.num_class, load_checkpoint_path=best_ckpt)

    if args_opt.do_predict==True:
        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file, do_lower_case=False)
        finetune_ckpt_url = args_opt.load_checkpoint_url
        if args_opt.do_eval==False:
            if finetune_ckpt_url.endswith(".ckpt") ==False:
                raise "For predict, if do_eval==False, you should select only one checkpoint file and this file should end with .ckpt"
            else:
                best_ckpt=finetune_ckpt_url
        do_predict(bert_net_cfg.seq_length, BertCLS, args_opt.num_class, load_checkpoint_path=best_ckpt,
                   tokenizer=tokenizer)

    if args_opt.do_attention==True:
        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file, do_lower_case=False)
        finetune_ckpt_url = args_opt.load_checkpoint_url
        if args_opt.do_eval==False:
            if finetune_ckpt_url.endswith(".ckpt") ==False:
                raise "For predict, if do_eval==False, you should select only one checkpoint file and this file should end with .ckpt"
            else:
                best_ckpt=finetune_ckpt_url
        do_attention(bert_net_cfg.seq_length, BertCLS, args_opt.num_class, load_checkpoint_path=best_ckpt,
                   tokenizer=tokenizer)

    print("FINISH !!!")

if __name__ == "__main__":
    print(args_opt)
    run_classifier()
