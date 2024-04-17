import os
from glob import glob
from argparse import ArgumentParser
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys

def parse_args():
    parser = ArgumentParser(description="Run train and predict on server 191")
    parser.add_argument("-E","--hmmer_dataset", type=str, required=True,help="HMM dataset path")
    parser.add_argument("-D","--device", type=str, required=True,help="ascend device id")

    parser.add_argument("-V","--vocab_file", type=str, required=True,help="vocab file path")
    parser.add_argument("-F","--fasta_file", type=str, required=True,help="your protein fasta file, only one seq")
    parser.add_argument("-P","--pretrain_model", type=str, required=True,help="pretrain model path")

    parser.add_argument("-X","--exp_path", type=str, required=True,help="MPB-EXP model path")
    parser.add_argument("-O","--output_path", type=str, required=True,help="output path")


    parser.add_argument( "--generate_num", type=str, required=False,default="100000", help="generate seq num")
    parser.add_argument( "--mask_prob", type=str, required=False,default="0.1", help="generate seq num")
    parser.add_argument( "--do_train", type=bool, required=False,default=True, help="run train?")
    parser.add_argument( "--do_generate", type=bool, required=False,default=True, help="run generates?")
    parser.add_argument( "--do_cls", type=bool, required=False,default=True, help="run cls?")
    parser.add_argument("-M","--model", type=str, required=False,help="cls model name")
    args_opt = parser.parse_args()
    return args_opt

def make_mindrecord(dataset_path,args):
    print("================")
    print("setp 1: make mindrecord dataset")
    print("input:",dataset_path)
    print("output:",dataset_path)
    print("================")

    #generate_mask_dataset
    cmd='python ' \
        './generate_dataset/generate_seq_for_mask.py --mask_prob '+args.mask_prob+' ' \
        '--data_dir '+dataset_path+' ' \
        '--vocab_file '+args.vocab_file+' ' \
        '--output_dir '+dataset_path+' ' \
        '--max_seq_length 1024 --do_train True --do_eval True --do_test True ' \
        '1> '+dataset_path+'/data_process_log.log 2> '+dataset_path+'/data_process_sys.log'
    print(cmd,flush=True)
    os.system(cmd)

def run_mask_train(dataset_path,device_id,args):

    print("================")
    print("setp 2: run train mask model")
    print("input:",dataset_path)
    print("output:",dataset_path)
    print("================")

    cmd='python ' \
        './model/mpbert_mask.py ' \
        '--config_path ./model/config_1024.yaml ' \
        '--do_train True ' \
        '--do_eval True ' \
        '--description sequence ' \
        '--epoch_num 200 ' \
        '--early_stopping_rounds 50 ' \
        '--frozen_bert False ' \
        '--device_id '+str(device_id)+' ' \
        '--data_url '+dataset_path+' ' \
        '--load_checkpoint_url '+args.pretrain_model+' ' \
        '--output_url '+dataset_path+' ' \
        '--task_name mask ' \
        '--train_batch_size 32 ' \

    print(cmd,flush=True)
    os.system(cmd)

def generate_mut(fasta_path,dataset_path,device_id,generate_num,args):


    model_path=dataset_path+"/mask_Best_Model.ckpt"
    csv_path=args.outptu_path
    if os.path.exists(csv_path)==False:
        os.makedirs(csv_path)

    print("================")
    print("setp 3: make mask from fasta and run predict")
    print("input:",fasta_path)
    print("output:",csv_path)
    print("================",flush=True)

    cmd='python ' \
        './model/mpbert_mask.py ' \
        '--config_path ./model/config_1024.yaml ' \
        "--vocab_file  "+args.vocab_file+" " \
        '--do_predict True ' \
        '--description sequence ' \
        '--device_id '+str(device_id)+' ' \
        '--data_url '+fasta_path+' ' \
        '--load_checkpoint_url '+model_path+' ' \
        '--output_url '+csv_path+' ' \
        '--predict_mask_num '+generate_num+' ' \
        '--mask_prob 0.1 '
    print(cmd,flush=True)
    os.system(cmd)

def extract_mask_result(fasta_path,args):
    fasta_name = os.path.basename(fasta_path).split(".")[0]
    csv_path=args.outptu_path
    pickle_file=csv_path+fasta_name+".pkl"

    print("================")
    print("setp 4: extract predict result")
    print("input:",pickle_file)
    print("output:",csv_path+fasta_name+".csv")
    print("================")

    pickle_res=pickle.load(open(pickle_file,"rb"))

    all_df=[]

    for pred_info in tqdm(pickle_res):
        all_df.append({
            "id":pred_info["seq_id"],
            "seq":"".join(pred_info["pred_seq"]),
            "mask_logits_mean":pred_info["logits"].max(axis=1).mean(),
            "mask_logits_min":pred_info["logits"].max(axis=1).min(),
            "label":-1,
            "is_select":None
        })
    all_df=pd.DataFrame(all_df)

    #sort mask_logits_min from high to low
    all_df=all_df.sort_values(by="mask_logits_min",ascending=False)

    wild_type_info={
        "id": pickle_res[0]["seq_id"].split("_")[0]+"_WT",
        "seq": "".join(pickle_res[0]["seq"]),
        "mask_logits_mean": -1,
        "mask_logits_min":-1,
        "label": 0,
        "is_select":True
    }
    wild_type_info=pd.DataFrame([wild_type_info])
    all_df=pd.concat([wild_type_info,all_df],axis=0)
    print(all_df)
    print(all_df.values.shape)

    all_df=all_df.drop_duplicates(subset=["seq"],keep="first")
    print(all_df.values.shape)

    #select top 100000
    all_df=all_df.iloc[:100000,:].copy(deep=True)
    print(all_df.values.shape)

    #select top 100
    select_df=all_df.iloc[:201,:].copy(deep=True)
    select_df["is_select"]=[True]*201

    all_df["is_select"]=[True]*201+[False]*(len(all_df)-201)

    print(select_df)
    print(all_df)

    select_df.to_csv(csv_path+fasta_name+".csv",index=False)
    all_df.to_csv(csv_path+fasta_name+"_all.csv",index=False)
    print(len(select_df))

def do_cls(model,fasta_path,device_id,args):
    if fasta_path.endswith(".fasta"):
        fasta_name = os.path.basename(fasta_path).split(".")[0]
        csv_path=args.outptu_path
        data_path = csv_path+fasta_name+".csv"
    elif fasta_path.endswith(".csv"):
        print("This is a csv file")
        fasta_name = os.path.basename(fasta_path).split(".")[0]
        csv_path = os.path.dirname(fasta_path)
        data_path = fasta_path
        print("data_path:",data_path)
        print("data_name:",fasta_name)
        print("data_dir:",csv_path)
    save_path = csv_path+"/" + model + "_predict/"

    print("================")
    print("setp 5: run cls")
    print("input:",csv_path)
    print("output:",save_path + fasta_name+"_predict_concat_" + model + ".csv")
    print("================",flush=True)

    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    cp_data_name_list=[]

    for cls_fold in range(5):
        cp_data_name=save_path + fasta_name+"_fold_" + str(cls_fold) + "." + data_path.split(".")[-1]
        cp_data_name_list.append(cp_data_name)

        print("copy",data_path,"to",cp_data_name)
        os.system("cp " + data_path + " " + cp_data_name)

        assert os.path.exists(cp_data_name)

        cmd = "python ./model/mpbert_classification.py" \
              " --config_path ./model/config_1024.yaml " \
              "--do_predict True --description classification " \
              "--num_class 2 " \
              "--vocab_file  "+args.vocab_file+" " \
              "--device_id " + device_id + " " \
              "--return_csv True " \
              "--return_sequence False " \
              "--data_url " + cp_data_name + " " \
              "--load_checkpoint_url "+args.exp_path+"/"+ model + "/fold_" + str(
            cls_fold) + ".ckpt " \
                    "--output_url " + save_path
        print(cmd)
        os.system(cmd)
        os.system("rm " + cp_data_name)
    return cp_data_name_list

def extract_cls_result(cp_data_name_list,model,fasta_path,args):
    if fasta_path.endswith(".fasta"):
        fasta_name = os.path.basename(fasta_path).split(".")[0]
        csv_path=args.outptu_path
    elif fasta_path.endswith(".csv"):
        print("This is a csv file")
        fasta_name = os.path.basename(fasta_path).split(".")[0]
        csv_path = os.path.dirname(fasta_path)
        data_path = fasta_path
        print("data_path:",data_path)
        print("data_name:",fasta_name)
        print("data_dir:",csv_path,flush=True)
    save_path = csv_path+"/" + model + "_predict/"

    res = []
    for i in range(5):
        f1 = pd.read_csv(cp_data_name_list[i].split(".csv")[0].split(".fasta")[0] + "_predict_result.csv")
        f1["fold"] = i
        res.append(f1)

    res = pd.concat(res)
    res = res.sort_values(by=["Unnamed: 0"])
    res.to_csv(save_path + fasta_name+"_predict_result_" + model + ".csv",
               index=False)

    df=pd.read_csv(save_path + fasta_name+"_predict_result_" + model + ".csv")

    print(df)

    header_list=["id","seq"]

    if "is_selset" in list(df.columns):
        header_list.append("is_selset")

    df=df.groupby(header_list, as_index=False)['dense'].mean()
    df=df.sort_values(by="dense",ascending=False)

    df.to_csv(save_path + fasta_name+"_predict_concat_" + model + ".csv",index=False)

def main():
    args = parse_args()
    print(args)
    dataset_path = args.hmmer_dataset
    device_id = args.device
    fasta_path = args.fasta_file

    if args.do_train:
        make_mindrecord(dataset_path,args)
        run_mask_train(dataset_path, device_id,  args)
    if args.do_generate:
        generate_mut(fasta_path, dataset_path, device_id,args.generate_num,args)
        extract_mask_result(fasta_path,args)
    if args.do_cls:
        if args.model==None:
            print("please input model name")
            sys.exit(0)
        elif args.model=="all":
            models=['547559', '64091', '880073', '189518', '246200', '208964', '160488', '99287', '511145', '272620', '1286170', '709991', '537011', '411477', '435591', '435590', '226186', '272559', '411479', '449447', '1140', '100226', '83332', '246196', '1280', '224308', '610130', '411902', '411470', '55529', '44689', '2903', '5691', '353153', '2850', '35128', '5811', '5833', '73239', '3055', '3218', '4577', '39947', '4565', '4513', '4097', '4081', '4113', '29760', '3880', '3847', '2711', '3635', '3702', '3708', '214684', '4896', '5507', '5141', '5476', '284590', '4932', '6239', '6945', '7460', '7227', '7165', '7159', '7955', '30732', '8030', '8022', '8364', '8355', '9031', '9612', '9615', '9796', '9823', '9913', '89462', '9544', '9598', '9606', '9986', '10029', '10090', '10116']
        else:
            models=args.model.split(",")
        for model in models:
            cp_data_name_list=do_cls(model, fasta_path, device_id,args)
            extract_cls_result(cp_data_name_list, model, fasta_path,args)



if __name__ == "__main__":
    main()
