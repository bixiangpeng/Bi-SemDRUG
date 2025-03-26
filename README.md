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

  🌳 The processed data can be downloaded through [link1](https://pan.baidu.com/s/1BAeE5P5mFJSAK02P5f223g?pwd=8aym) and [link2](https://pan.baidu.com/s/1XLWvZQATrfXagoX_7Ey7yg?pwd=vvb7).
  
  🌳 If you want to re-partition the subgraph, you can simply execute `python data/subgraph_partitioning.py `.


* ### Training
  You can retrain the model from scratch with the following command:
  ```text
  For `DrugBank` dataset:
    python main.py --dataset DrugBank --ddi_types 86 --rst_file ./model_pkl/DrugBank/ --folds 3 --epochs 200 --hidden_dim 128 --hyper_num 4 

  For `TwoSides` dataset:
    python main.py --dataset TwoSides --ddi_types 963 --rst_file ./model_pkl/TwoSides/ --folds 3 --epochs 200 --hidden_dim 128 --hyper_num 4 

  For `DeepDDI` dataset:
    python main.py --dataset DeepDDI --ddi_types 2 --rst_file ./model_pkl/DeepDDI/ --folds 3 --epochs 200 --hidden_dim 128 --hyper_num 4 

   ```
  
  Here is the detailed introduction of the optional parameters when running `main_training.py`:
   ```text
    --datasetname: The dataset name, specifying the dataset used for model training.
    --ddi_types: The number of interaction types between drugs in the dataset..
    --hidden_dim: The dimension of node embedding in hierarchical knowledge graph.
    --hyper_num: The number of hyperedges learned in the HHGNN network.
    --rst_file: The storage location of the model checkpoint file..
    --device_id: The device, specifying the GPU device number used for training.
    --batch_size: The batch size, specifying the number of samples in each training batch.
    --epochs: The number of epochs, specifying the number of iterations for training the model on the entire dataset.
    --lr: The learning rate, controlling the rate at which model parameters are updated.
    --num_workers: This parameter is an optional value in the Dataloader, and when its value is greater than 0, it enables multiprocessing for data processing.
   ```

## Contact

We welcome you to contact us (email: bixiangpeng@stu.ouc.edu.cn) for any questions and cooperations.

