# MPB-EXP-MUT

MPB-EXP and MPB-MUT models are two key models developed in this study, which use deep learning techniques to predict and optimize the soluble expression levels of proteins.

![MPB-EXP-MUT]("./Asset/Figure 1.jpg)

### MPB-EXP Model

The MPB-EXP model is a classification model based on amino acid sequences, used to predict the soluble expression levels of proteins in different host organisms. This model employs a transfer learning approach, initially constructing a pre-trained model (MP-TRANS) containing multiple Transformer layers, which can capture complex features of protein sequences. It is then fine-tuned using specific protein expression level datasets to create 88 MPB-EXP models for 88 different species. Each model can predict the expression levels of proteins within its corresponding species, helping to identify key sequence features that influence protein expression by comparing amino acid sequences with expression data.

### MPB-MUT Model

The MPB-MUT model is a mutation generation model further developed from the same transfer-learning pre-trained MP-TRANS model. Using the knowledge gained during pre-training, this model designs targeted mutations in specific protein sequences to generate mutants that might enhance protein expression levels. In practice, the model predicts the most likely beneficial alternative residues by randomly masking amino acids in the sequence. The generated mutants are then evaluated for their potential expression in specific hosts (such as *E. coli*) and validated experimentally for expression and activity.

The combined use of these two models not only predicts protein expression levels but also directly participates in the optimization design of protein sequences, greatly improving the efficiency and breadth of biotechnological applications. This method eliminates the need to rely on traditional biological information (such as the three-dimensional structure of proteins), significantly simplifying the complexity of protein engineering and allowing researchers to design and modify proteins more directly from sequence information.

### Before using

#### Sequence alignment

For sequence alignment, use the HMMER Search local tool. You should download it from the official [HMMER](hmmer.org) website or install it using conda:

```shell
conda install hmmer -c bioconda
```

After installation, use `jackhmmer -h` to check if it has been installed successfully.

Next, you should download the latest version of the UniRef90 fasta file from the official [UniRef](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz) FTP site.

Please store the downloaded UniRef90 file in a separate folder. The UniRef90 file contains a large number of sequences, and to speed up the execution of our script, you might consider splitting it into multiple sub-fasta files and saving them in the same folder.

In addition, you need to install the following dependency packages:

```
pip install numpy scikit-learn tqdm pandas
```

#### Expression level prediction and mutant design

Our model operates on the [![](https://img.shields.io/badge/Framework-mindspore=1.8-blue.svg??style=flat-square)](https://www.mindspore.cn/en) and can be trained and inferred on GPUs and Huawei Ascend NPUs. For GPU execution, we recommend using Conda. For running on Huawei Ascend NPUs, we recommend using Docker or Conda.

Link to download MindSpore:

| Device | Conda                                                        | Docker                                                       | pip                                                          |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GPU    | [mindspore_gpu_install_conda_en.md](https://gitee.com/mindspore/docs/blob/r1.8/install/mindspore_gpu_install_conda_en.md) | [mindspore_gpu_install_docker_en.md](https://gitee.com/mindspore/docs/blob/r1.8/install/mindspore_gpu_install_docker_en.md) | [mindspore_gpu_install_pip_en.md](https://gitee.com/mindspore/docs/blob/r1.8/install/mindspore_gpu_install_pip_en.md) |
| Ascend | [mindspore_ascend_install_conda_en.md](https://gitee.com/mindspore/docs/blob/r1.8/install/mindspore_ascend_install_conda_en.md) | [mindspore_ascend_install_docker_en.md](https://gitee.com/mindspore/docs/blob/r1.8/install/mindspore_ascend_install_docker_en.md) | [mindspore_ascend_install_pip_en.md](https://gitee.com/mindspore/docs/blob/r1.8/install/mindspore_ascend_install_pip_en.md) |

In addition, you need to install the following dependency packages:

```
pip install scikit-learn tqdm pandas matplotlib
```

#### Model and data download

We have uploaded all models and data to Zenodo. First, you need to download the relevant weights for our MP-TRANS pretrained model:

| Model    | link |
| -------- | ---- |
| MP-TRANS |      |

Next, you need to download our MPB-EXP models for 88 species. Due to Zenodo's storage limits, we have split them into 9 parts, with each link containing a different subset of species. Please refer to the details on Zenodo for specifics about which species are included:

| Model           | link |
| --------------- | ---- |
| MPB_EXP_model_0 |      |
| MPB_EXP_model_1 |      |
| MPB_EXP_model_2 |      |
| MPB_EXP_model_3 |      |
| MPB_EXP_model_4 |      |
| MPB_EXP_model_5 |      |
| MPB_EXP_model_6 |      |
| MPB_EXP_model_7 |      |
| MPB_EXP_model_8 |      |

In addition, we also provide the original training data for MPB-EXP:

| Dataset                                                  | link |
| -------------------------------------------------------- | ---- |
| Protein expression level dataset (MPB-EXP model dataset) |      |



### Mutant Design

We have encapsulated all the steps in two Python scripts, and the specific usage steps are as follows:

#### run_sequence_align

This script is used to construct the training data for the mutant design model MPB-MUT through HMMER sequence alignment. The key parameters are defined as follows:

| Parameter      | Short flag | Expansion                                                    |
| -------------- | ---------- | ------------------------------------------------------------ |
| --fasta_file   | -F         | This fasta file should contain only one sequence, the sequence you wish to mutate. |
| --output       | -O         | Output path.                                                 |
| --uniref       | -U         | Fasta file of UniRef90, please ensure you have decompressed this file. |
| --uniref_split | -S         | This path should store the multiple sub-fasta files into which you have split the UniRef90 file to accelerate script execution. If not defined here, the path for --UniRef90 will be used. |

You can run this script using the following command. Once the run is complete, you will find three files: train, val, test, in the path set by `--output`.

```shell
python run_sequence_align.py -F [Fasta file of your protein] -O [Output path] -U [Fasta file of UniRef90] -S [Path of sub-Uniref90 fasta]
```

#### mut_generation_flow

Next, use this script to generate mutants. The main parameters are as follows:

| Parameter        | Short flag | Expansion                                                    |
| ---------------- | ---------- | ------------------------------------------------------------ |
| --hmmer_dataset  | -E         | The output path from the previous step, which stores the train, val, and test files created by the run_sequence_align.py script. |
| --device         | -D         | Device ID for training, enter the ID of the card to be used when multiple deep learning training cards are available, enter 0 when only one card is present. |
| --vocab_file     | -V         | Transformer vocab dict file, you can see this file in [here](./model/vocab_v2.txt). |
| --fasta_file     | -F         | This fasta file should contain only one sequence, the sequence you want to mutate. |
| --pretrain_model | -P         | Pre-trained weights ckpt file, which you need to download from Zenodo. |
| --exp_path       | -X         | Path to the MPB-EXP model files, this path should contain multiple folders, each named after the TaxonID of a specific species. You need to download this from Zenodo. |
| --output_path    | -O         | Output path.                                                 |
| --model          | -M         | The MPB-EXP model you wish to use, please enter the TaxonID of the corresponding species. |

You can run this script with the following code, and once it completes, you will find a file named *_concat_result.csv in the output directory. After sorting this file by the Dense value from highest to lowest, this value will represent the likelihood of high expression probability. Select the mutant sequences with high Dense values as the final result.

```shell
python mut_generation_flow.py -E [HMMER dataset path] -D [GPU/NPU device ID] -V [Vocab file] -F [Fasta file of your protein] -P [MP-Trans file] -X [MPB-EXP path] -O [Output path] -M [TaxonID]
```

### Related article for this code:

