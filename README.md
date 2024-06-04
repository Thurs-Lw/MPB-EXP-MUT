# MPB-EXP-MUT

MPB-EXP and MPB-MUT models are two key models developed in this study, which use deep learning techniques to predict and optimize the soluble expression levels of proteins.

<img src="./Asset/Figure 1.jpg"  />

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

We have uploaded all models and data to Zenodo. 

1. **Download the relevant weights for our MP-TRANS pretrained model for mutant generation:**

| Model    | link                                               |
| -------- | -------------------------------------------------- |
| MP-TRANS | [Zenodo]([https://doi.org/10.5281/zenodo.11191176) |

2. **Download our MPB-EXP models of 88 species for expression level prediction:**

| Archive | Model                                         | ID      | link                                               |
| ------- | --------------------------------------------- | ------- | -------------------------------------------------- |
| 0       | *Natrialba magadii*                           | 547559  | [Zenodo]([https://doi.org/10.5281/zenodo.10985361) |
| 0       | *Halobacterium salinarum*                     | 64091   | [Zenodo]([https://doi.org/10.5281/zenodo.10985361) |
| 0       | *Caldithrix abyssi*                           | 880073  | [Zenodo]([https://doi.org/10.5281/zenodo.10985361) |
| 0       | *Leptospira interrogans*                      | 189518  | [Zenodo]([https://doi.org/10.5281/zenodo.10985361) |
| 0       | *Ruegeria pomeroyi*                           | 246200  | [Zenodo]([https://doi.org/10.5281/zenodo.10985361) |
| 0       | *Pseudomonas aeruginosa*                      | 208964  | [Zenodo]([https://doi.org/10.5281/zenodo.10985361) |
| 0       | *Pseudomonas putida*                          | 160488  | [Zenodo]([https://doi.org/10.5281/zenodo.10985361) |
| 0       | *Salmonella enterica*                         | 99287   | [Zenodo]([https://doi.org/10.5281/zenodo.10985361) |
| 0       | *Escherichia coli*                            | 511145  | [Zenodo]([https://doi.org/10.5281/zenodo.10985361) |
| 0       | *Klebsiella pneumoniae*                       | 272620  | [Zenodo]([https://doi.org/10.5281/zenodo.10985361) |
| 1       | *Raoultella ornithinolytica*                  | 1286170 | [Zenodo]([https://doi.org/10.5281/zenodo.10985419) |
| 1       | *Odoribacter splanchnicus*                    | 709991  | [Zenodo]([https://doi.org/10.5281/zenodo.10985419) |
| 1       | *Segatella copri*                             | 537011  | [Zenodo]([https://doi.org/10.5281/zenodo.10985419) |
| 1       | *Parabacteroides merdae*                      | 411477  | [Zenodo]([https://doi.org/10.5281/zenodo.10985419) |
| 1       | *Parabacteroides distasonis*                  | 435591  | [Zenodo]([https://doi.org/10.5281/zenodo.10985419) |
| 1       | *Phocaeicola vulgatus*                        | 435590  | [Zenodo]([https://doi.org/10.5281/zenodo.10985419) |
| 1       | *Bacteroides thetaiotaomicron*                | 226186  | [Zenodo]([https://doi.org/10.5281/zenodo.10985419) |
| 1       | *Bacteroides fragilis*                        | 272559  | [Zenodo]([https://doi.org/10.5281/zenodo.10985419) |
| 1       | *Bacteroides uniformis*                       | 411479  | [Zenodo]([https://doi.org/10.5281/zenodo.10985419) |
| 1       | *Microcystis aeruginosa*                      | 449447  | [Zenodo]([https://doi.org/10.5281/zenodo.10985419) |
| 2       | *Synechococcus elongatus*                     | 1140    | [Zenodo]([https://doi.org/10.5281/zenodo.10985453) |
| 2       | *Streptomyces coelicolor*                     | 100226  | [Zenodo]([https://doi.org/10.5281/zenodo.10985453) |
| 2       | *Mycobacterium tuberculosis*                  | 83332   | [Zenodo]([https://doi.org/10.5281/zenodo.10985453) |
| 2       | *Mycolicibacterium smegmatis*                 | 246196  | [Zenodo]([https://doi.org/10.5281/zenodo.10985453) |
| 2       | *Staphylococcus aureus*                       | 1280    | [Zenodo]([https://doi.org/10.5281/zenodo.10985453) |
| 2       | *Bacillus subtilis*                           | 224308  | [Zenodo]([https://doi.org/10.5281/zenodo.10985453) |
| 2       | *Clostridium saccharolyticum*                 | 610130  | [Zenodo]([https://doi.org/10.5281/zenodo.10985453) |
| 2       | *Enterocloster bolteae*                       | 411902  | [Zenodo]([https://doi.org/10.5281/zenodo.10985453) |
| 2       | *Ruminococcus gnavus*                         | 411470  | [Zenodo]([https://doi.org/10.5281/zenodo.10985453) |
| 2       | *Guillardia theta*                            | 55529   | [Zenodo]([https://doi.org/10.5281/zenodo.10985453) |
| 3       | Dictyostelium discoideum                      | 44689   | [Zenodo]([https://doi.org/10.5281/zenodo.10985479) |
| 3       | Emiliania huxleyi                             | 2903    | [Zenodo]([https://doi.org/10.5281/zenodo.10985479) |
| 3       | Trypanosoma brucei                            | 5691    | [Zenodo]([https://doi.org/10.5281/zenodo.10985479) |
| 3       | Trypanosoma cruzi strain CL Brener            | 353153  | [Zenodo]([https://doi.org/10.5281/zenodo.10985479) |
| 3       | Phaeodactylum tricornutum                     | 2850    | [Zenodo]([https://doi.org/10.5281/zenodo.10985479) |
| 3       | Thalassiosira pseudonana                      | 35128   | [Zenodo]([https://doi.org/10.5281/zenodo.10985479) |
| 3       | Toxoplasma gondi                              | 5811    | [Zenodo]([https://doi.org/10.5281/zenodo.10985479) |
| 3       | Plasmodium falciparum                         | 5833    | [Zenodo]([https://doi.org/10.5281/zenodo.10985479) |
| 3       | Plasmodium yoelii yoelii                      | 73239   | [Zenodo]([https://doi.org/10.5281/zenodo.10985479) |
| 3       | Chlamydomonas reinhardtii                     | 3055    | [Zenodo]([https://doi.org/10.5281/zenodo.10985479) |
| 4       | Physcomitrium patens                          | 3218    | [Zenodo]([https://doi.org/10.5281/zenodo.10985483) |
| 4       | Zea mays                                      | 4577    | [Zenodo]([https://doi.org/10.5281/zenodo.10985483) |
| 4       | Oryza sativa Japonica Group                   | 39947   | [Zenodo]([https://doi.org/10.5281/zenodo.10985483) |
| 4       | Triticum aestivum                             | 4565    | [Zenodo]([https://doi.org/10.5281/zenodo.10985483) |
| 4       | Hordeum vulgare                               | 4513    | [Zenodo]([https://doi.org/10.5281/zenodo.10985483) |
| 4       | Nicotiana tabacum                             | 4097    | [Zenodo]([https://doi.org/10.5281/zenodo.10985483) |
| 4       | Solanum lycopersicum                          | 4081    | [Zenodo]([https://doi.org/10.5281/zenodo.10985483) |
| 4       | Solanum tuberosum                             | 4113    | [Zenodo]([https://doi.org/10.5281/zenodo.10985483) |
| 4       | Vitis vinifera                                | 29760   | [Zenodo]([https://doi.org/10.5281/zenodo.10985483) |
| 4       | Medicago truncatula                           | 3880    | [Zenodo]([https://doi.org/10.5281/zenodo.10985483) |
| 5       | Glycine max                                   | 3847    | [Zenodo]([https://doi.org/10.5281/zenodo.10985489) |
| 5       | Citrus sinensis                               | 2711    | [Zenodo]([https://doi.org/10.5281/zenodo.10985489) |
| 5       | Gossypium hirsutum                            | 3635    | [Zenodo]([https://doi.org/10.5281/zenodo.10985489) |
| 5       | Arabidopsis thaliana                          | 3702    | [Zenodo]([https://doi.org/10.5281/zenodo.10985489) |
| 5       | Brassica napus                                | 3708    | [Zenodo]([https://doi.org/10.5281/zenodo.10985489) |
| 5       | Cryptococcus neoformans var. neoformans JEC21 | 214684  | [Zenodo]([https://doi.org/10.5281/zenodo.10985489) |
| 5       | Schizosaccharomyces pombe                     | 4896    | [Zenodo]([https://doi.org/10.5281/zenodo.10985489) |
| 5       | Fusarium oxysporum                            | 5507    | [Zenodo]([https://doi.org/10.5281/zenodo.10985489) |
| 5       | Neurospora crassa                             | 5141    | [Zenodo]([https://doi.org/10.5281/zenodo.10985489) |
| 5       | Candida albicans                              | 5476    | [Zenodo]([https://doi.org/10.5281/zenodo.10985489) |
| 6       | Kluyveromyces lactis NRRL Y-1140              | 284590  | [Zenodo]([https://doi.org/10.5281/zenodo.10985495) |
| 6       | Saccharomyces cerevisiae                      | 4932    | [Zenodo]([https://doi.org/10.5281/zenodo.10985495) |
| 6       | Caenorhabditis elegans                        | 6239    | [Zenodo]([https://doi.org/10.5281/zenodo.10985495) |
| 6       | Ixodes scapularis                             | 6945    | [Zenodo]([https://doi.org/10.5281/zenodo.10985495) |
| 6       | Apis mellifera                                | 7460    | [Zenodo]([https://doi.org/10.5281/zenodo.10985495) |
| 6       | Drosophila melanogaster                       | 7227    | [Zenodo]([https://doi.org/10.5281/zenodo.10985495) |
| 6       | Anopheles gambiae                             | 7165    | [Zenodo]([https://doi.org/10.5281/zenodo.10985495) |
| 6       | Aedes aegypti                                 | 7159    | [Zenodo]([https://doi.org/10.5281/zenodo.10985495) |
| 6       | Danio rerio                                   | 7955    | [Zenodo]([https://doi.org/10.5281/zenodo.10985495) |
| 6       | Oryzias melastigma                            | 30732   | [Zenodo]([https://doi.org/10.5281/zenodo.10985495) |
| 7       | Salmo salar                                   | 8030    | [Zenodo]([https://doi.org/10.5281/zenodo.10985513) |
| 7       | Oncorhynchus mykiss                           | 8022    | [Zenodo]([https://doi.org/10.5281/zenodo.10985513) |
| 7       | Xenopus tropicalis                            | 8364    | [Zenodo]([https://doi.org/10.5281/zenodo.10985513) |
| 7       | Xenopus laevis                                | 8355    | [Zenodo]([https://doi.org/10.5281/zenodo.10985513) |
| 7       | Gallus gallus                                 | 9031    | [Zenodo]([https://doi.org/10.5281/zenodo.10985513) |
| 7       | Canis lupus                                   | 9612    | [Zenodo]([https://doi.org/10.5281/zenodo.10985513) |
| 7       | Canis lupus(familiaris)                       | 9615    | [Zenodo]([https://doi.org/10.5281/zenodo.10985513) |
| 7       | Equus caballus                                | 9796    | [Zenodo]([https://doi.org/10.5281/zenodo.10985513) |
| 7       | Sus scrofa                                    | 9823    | [Zenodo]([https://doi.org/10.5281/zenodo.10985513) |
| 7       | Bos taurus                                    | 9913    | [Zenodo]([https://doi.org/10.5281/zenodo.10985513) |
| 8       | Bubalus bubalis                               | 89462   | [Zenodo]([https://doi.org/10.5281/zenodo.10985522) |
| 8       | Macaca mulatta                                | 9544    | [Zenodo]([https://doi.org/10.5281/zenodo.10985522) |
| 8       | Pan troglodytes                               | 9598    | [Zenodo]([https://doi.org/10.5281/zenodo.10985522) |
| 8       | Homo sapiens                                  | 9606    | [Zenodo]([https://doi.org/10.5281/zenodo.10985522) |
| 8       | Oryctolagus cuniculus                         | 9986    | [Zenodo]([https://doi.org/10.5281/zenodo.10985522) |
| 8       | Cricetulus griseus                            | 10029   | [Zenodo]([https://doi.org/10.5281/zenodo.10985522) |
| 8       | Mus musculus                                  | 10090   | [Zenodo]([https://doi.org/10.5281/zenodo.10985522) |
| 8       | Rattus norvegicus                             | 10116   | [Zenodo]([https://doi.org/10.5281/zenodo.10985522) |

3. In addition, we also provide the original training data for MPB-EXP:

| Dataset                                                  | link                                               |
| -------------------------------------------------------- | -------------------------------------------------- |
| Protein expression level dataset (MPB-EXP model dataset) | [Zenodo]([https://doi.org/10.5281/zenodo.10984375) |



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

