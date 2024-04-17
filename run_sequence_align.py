from argparse import ArgumentParser
from tqdm import tqdm
import os
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from glob import glob

def parse_args():
    parser = ArgumentParser(description="Run HMM to find homologous sequences on server 192")
    parser.add_argument("-T","--threshold", type=float, default=0.5,help="set hmmer threshold from 0-1(0-100%)")
    parser.add_argument("-F","--fasta_file", type=str, required=True,help="your protein fasta file, only one seq")
    parser.add_argument("-O","--output",type=str, required=False,default=None)
    parser.add_argument("-U","--uniref",type=str, required=True,help="uniref90 file")
    parser.add_argument("-S","--uniref_split",type=str, required=False,default=None,
                        help="uniref90 file split dir, this dir should contain fasta files that split from uniref90 file, to speed up the search process.")

    args_opt = parser.parse_args()
    return args_opt

def read_fasta(fasta_file):
    seq=[]
    with open(fasta_file) as f:
        for line in f:
            line=line.strip()
            if line.startswith(">"):
                seq.append([line[1:], ""])
            else:
                seq[-1][-1]+=line
    return seq

def run_hmmer(input_fasta,input_name,output_dir,args):
    protein=read_fasta(input_fasta)
    assert len(protein)==1
    seq_len=len(protein[0][1])
    seq_threshold=int(seq_len*args.threshold)
    print("input_fasta:",input_fasta)
    print("seq_len:",seq_len)
    print("seq_threshold:",seq_threshold)
    print("run hmmer",flush=True)
    hmmer_script=[  "jackhmmer",
                    "-N","5",
                    "-o",os.path.join(output_dir,input_name+".hmmer.out.o"),
                    "--tblout",os.path.join(output_dir,input_name+".hmmer.tblout"),
                    "--domtblout",os.path.join(output_dir,input_name+".hmmer.domtblout"),
                    "--notextw",
                    "-T",str(seq_threshold),
                    "--domT",str(seq_threshold),
                    "--incT",str(seq_threshold),
                    "--incdomT",str(seq_threshold),
                    "--cpu","50",
                    input_fasta,
                    args.uniref
                  ]
    hmmer_script=" ".join(hmmer_script)
    print(hmmer_script,flush=True)
    os.system(hmmer_script)
    assert os.path.exists(os.path.join(output_dir,input_name+".hmmer.domtblout"))


def read_tblout(tblout_file):
    print("reading tblout file:",tblout_file,flush=True)
    with open(tblout_file) as f:
        lines = f.readlines()
        unifre90_id = []
        align_value=[]
        for line in tqdm(lines):
            line = line.strip()
            if line.startswith("#"):
                continue
            line = line.split()
            seq_id = line[0]
            score = float(line[5])
            unifre90_id.append(seq_id)
            align_value.append(score)
        unifre90_id = list(set(unifre90_id))

        #sort uniref90 id by align value
        unifre90_id=[x for _,x in sorted(zip(align_value,unifre90_id),reverse=True)]
        unifre90_id=unifre90_id[:10000]

        print("Find uniref90 id num :", len(unifre90_id),flush=True)
    return unifre90_id

def extract_tblout(input_name,output_dir,args):
    tblout_file=os.path.join(output_dir,input_name+".hmmer.tblout")
    uniref_id=read_tblout(tblout_file)

    seq_result=[]

    if args.uniref_split is not None:
        args.uniref_split = os.path.dirname(args.uniref)

        if len(glob(os.path.join(args.uniref_split,"*.fasta")))!=1:
            raise Exception("uniref90 dir should contain only one fasta file")

    if len(glob(os.path.join(args.uniref_split,"*.fasta")))==0:
        raise Exception("uniref90 dir should contain at least one fasta file")

    print("uniref90 file :")
    print(glob(os.path.join(args.uniref_split,"*.fasta")))
    for uniref90_file in glob(os.path.join(args.uniref_split,"*.fasta")):
        print("reading uniref90 file:",uniref90_file,flush=True)
        uniref90_file = open(uniref90_file, "r").readlines()
        uniref_90_dict={}
        for i in tqdm(range(int(len(uniref90_file)/2))):
            uniref_90_dict[uniref90_file[i*2].strip().strip(">").split()[0]]=uniref90_file[i*2+1].strip()
        remove_list=[]

        for seq_id in tqdm(uniref_id):
            if seq_id in uniref_90_dict.keys():
                seq_result.append([seq_id,uniref_90_dict[seq_id]])
                remove_list.append(seq_id)
        for seq_id in remove_list:
            uniref_id.remove(seq_id)
        print("Find seq num in uniref90:", len(seq_result), "left seq num:", len(uniref_id), flush=True)

    print("Find seq num in uniref90:", len(seq_result), flush=True)
    return seq_result

def write_fasta(seq_list,output_dir,input_name):
    print("write fasta file",flush=True)
    with open(os.path.join(output_dir,input_name+".hmmer.fasta"),"w") as f:
        for seq in seq_list:
            f.write(">"+seq[0]+"\n")
            f.write(seq[1]+"\n")

def make_dataset(output_dir,input_name,args):
    HMM_results = os.path.join(output_dir,input_name+".hmmer.fasta")
    hmm_data = read_fasta(HMM_results)
    assert len(hmm_data) == len(set([i[1] for i in hmm_data]))
    hmm_data = np.array(hmm_data)
    temp, test = train_test_split(hmm_data, test_size=0.2, random_state=42)
    train,val=train_test_split(temp,test_size=0.2,random_state=42)


    save_path = os.path.join(output_dir,"HMM_"+input_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + "/train.fasta", "w") as f:
        [f.write(">" + i[0] + "\n" + i[1] + "\n") for i in train]
    with open(save_path + "/val.fasta", "w") as f:
        [f.write(">" + i[0] + "\n" + i[1] + "\n") for i in val]
    with open(save_path + "/test.fasta", "w") as f:
        [f.write(">" + i[0] + "\n" + i[1] + "\n") for i in test]
    print("This dir: ",save_path," contains train, val, test fasta files, is the --hmmer_dataset used in mut_generation_flow.py")
    print("Next, you can use mut_generation_flow.py to generate mutation dataset.")


if __name__ == "__main__":
    args = parse_args()
    input_dir=args.fasta_file
    input_file_name=os.path.basename(input_dir).split(".")[0]
    if args.output is None:
        output_dir=os.path.dirname(input_dir)
    else:
        output_dir=args.output
    run_hmmer(input_dir,input_file_name,output_dir,args)
    uniref90_result=extract_tblout(input_file_name,output_dir,args)
    write_fasta(uniref90_result,output_dir,input_file_name)
    make_dataset(output_dir,input_file_name,args)




