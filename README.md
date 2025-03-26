# Bi-SemDRUG
---
A repo for "Subgraph-focused Biomedical Knowledge Embedding with Bi-semantic Encoder for Multi-type Drug-drug Interaction Prediction ".

## Contents

* [Architecture](#Architecture)
* [Requirements](#requirements)
   * [Download projects](#download-projects)
   * [Configure the environment manually](#configure-the-environment-manually)
* [Usages](#usages)
   * [Data preparation](#data-preparation)
   * [Training](#training)
   * [Pretrained models](#pretrained-models)
* [Contact](#contact)

## Architecture
![Bi-SemDRUG architecture](https://github.com/bixiangpeng/Bi-SemDRUG/blob/main/architecture.png)

## Requirements

* ### Download projects

   Download the GitHub repo of this project onto your local server: `git clone https://github.com/bixiangpeng/Bi-SemDRUG/`


* ### Configure the environment manually

   Create and activate virtual env: `conda create -n Bi-SemDRUG python=3.7.12 ` and `conda activate Bi-SemDRUG`
   
   Install specified version of pytorch: `conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge`
   
   Install specified version of PyG: `pip install torch-geometric==2.0.2`
   
   :bulb: Note that the operating system we used is `ubuntu 22.04` and the version of Anaconda is `24.1.2`.

  
##  Usages

* ### Data preparation
  There are three benchmark datasets were adopted in this project, including `DrugBank`, `TwoSides`, and `DeepDDI`.

  🌳 The processed data can be downloaded through [this link](https://pan.baidu.com/s/1BAeE5P5mFJSAK02P5f223g?pwd=8aym).
  
  🌳 If you want to re-partition the subgraph, you can simply execute `python data/subgraph_partitioning.py `.


* ### Training
  You can retrain the model from scratch with the following command:
  ```text
  For `DIP S. cerevisiae` dataset:
    python main_training.py --datasetname DIP_S.cerevisiae --super_ratio 0.2 --layers 8 --hidden_dim 64

  For `STRING H. sapiens` dataset:
    python main_training.py --datasetname STRING_H.sapiens --super_ratio 0.2 --layers 8 --hidden_dim 64

  For `STRING S. cerevisiae` dataset:
    python main_training.py --datasetname STRING_S.cerevisiae --super_ratio 0.2 --layers 8 --hidden_dim 64

   ```
  
  Here is the detailed introduction of the optional parameters when running `main_training.py`:
   ```text
    --datasetname: The dataset name, specifying the dataset used for model training.
    --hidden_dim: The dimension of node embedding in hierarchical knowledge graph.
    --layers: The hop of HetSemGNN in semantic encoder.
    --super_ratio: The ratio of super-node used to generate graph context vector.
    --device_id: The device, specifying the GPU device number used for training.
    --batch_size: The batch size, specifying the number of samples in each training batch.
    --epochs: The number of epochs, specifying the number of iterations for training the model on the entire dataset.
    --lr: The learning rate, controlling the rate at which model parameters are updated.
    --num_workers: This parameter is an optional value in the Dataloader, and when its value is greater than 0, it enables multiprocessing for data processing.
   ```

## Contact

We welcome you to contact us (email: bixiangpeng@stu.ouc.edu.cn) for any questions and cooperations.

