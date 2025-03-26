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

## Abstracts
Discovering the inherent routine of interactions between proteins is of great importance in deciphering microscopic life systems. It is essentially a pattern recognition task since the most crucial step is to uncover the complex interaction patterns hidden in the biological knowledge graphs. Recent advances have shown great promise in this regard; however, existing solutions still overlook three critical issues: 1) category heterogeneity, 2) relation heterogeneity, and 3) annotation scarcity, which severely hinder the comprehensive identification and understanding of protein interaction patterns. To address these issues, we introduce a protein interaction pattern recognition framework based on heterogeneous semantics mining and hierarchical graph representation, namely SemanGraphPPI. Our model integrates the annotation knowledge graph with the interaction knowledge graph through hierarchical graph representation, enabling end-to-end function representation of proteins. Additionally, it effectively mines heterogeneous function semantics of proteins by explicitly modeling the heterogeneous information in the annotation knowl
edge graph, and enhances the function representation of *under-labeled* proteins through a well-designed hierarchical knowledge enhancement module. Exhaustive experiments on three bench mark datasets demonstrate that our proposed model achieves state-of-the-art performance compared to other baseline methods and exhibits good generalization and efficiency in large-scale PPI prediction.
![SemanGraphPPI architecture](https://github.com/bixiangpeng/SemanGraphPPI/blob/main/framework.png)

## Requirements

* ### Download projects

   Download the GitHub repo of this project onto your local server: `git clone https://github.com/bixiangpeng/SemanGraphPPI/`


* ### Configure the environment manually

   Create and activate virtual env: `conda create -n SemanGraphPPI python=3.8 ` and `conda activate SemanGraphPPI`
   
   Install specified version of pytorch: ` conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
   
   Install specified version of PyG: ` conda install pyg==2.5.2 -c pyg`
   
   :bulb: Note that the operating system we used is `ubuntu 22.04` and the version of Anaconda is `24.1.2`.

  
##  Usages

* ### Data preparation
  There are three benchmark datasets were adopted in this project, including `DIP S. cerevisiae`, `STRING S. cerevisiae`, and `STRING H. sapiens`.

  🌳 To facilitate understanding, the following is an explanation of the `data` file directory:
    ```text
       >  data
          ├── Annotation_KG               - A folder for annotation knowledge graph data.           
          │   ├── BP_subgraph.pkl             - A file for BP(Biological Process) knowledge subgraph.
          │   ├── MF_subgraph.pkl             - A file for MF(Molecular Function) knowledge subgraph.
          │   └── CC_subgraph.pkl             - A file for CC(Cellular Component) knowledge subgraph.
          ├── DIP_S.cerevisiae            - A folder for DIP S. cerevisiae dataset.
          │   ├── train.tsv                   - A TSV file for training dataset. 
          │   ├── test.tsv                    - A TSV file for test dataset. 
          │   ├── Interaction_KG              - A folder for interaction knowledge graph data.
          │       ├── IKG_edge.pkl                - A pkl file recording the edges in interaction knowledge graph.
          │       ├── edge_index_map_dict.pkl     - A pkl file recording the index of interaction edges in interaction knowledge graph.
          │       ├── index_map_dict.pkl          - A pkl file recording the index of protein nodes in interaction knowledge graph.
          │       ├── annotation_index_map.pkl    
          │       └── annotation_batch.pkl        
          ├── STRING_H.sapiens            - A folder for STRING H. sapiens dataset.
          └── STRING_S.cerevisiae         - A folder for STRING S. cerevisiae dataset.

   ```


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

* ### Pretrained models

   If you don't want to re-train the model, we provide pre-trained model parameters as shown below. 
<a name="pretrained-models"></a>
   | Datasets | Pre-trained models          | Description |
   |:-----------:|:-----------------------------:|:--------------|
   | DIP_S.cerevisiae    | [model](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/DIP_S.cerevisiae/pretrained_model.pkl) | The pretrained model parameters on the DIP S. cerevisiae dataset. |
   | STRING_S.cerevisiae     | [model](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/STRING_S.cerevisiae/pretrained_model.pkl) | The Pretrained model parameters on the STRING S. cerevisiae dataset. |
   | STRING_H.sapiens    | [model](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/STRING_H.sapiens/pretrained_model.pkl)   | The pretrained model parameters on the STRING H. sapiens dataset. |
  
   Based on these pre-trained models, you can perform PPI predictions by simply running the following command:
   ```text
    For DIP S. cerevisiae dataset:
      python inference.py --datasetname DIP_S.cerevisiae --super_ratio 0.2 --layers 8 --hidden_dim 64
  
    For STRING S. cerevisiae dataset:
      python inference.py --datasetname STRING_S.cerevisiae --super_ratio 0.2 --layers 8 --hidden_dim 64
  
    For STRING H. sapiens dataset:
      python inference.py --datasetname STRING_H.sapiens --super_ratio 0.2 --layers 8 --hidden_dim 64

   ```

## Contact

We welcome you to contact us (email: bixiangpeng@stu.ouc.edu.cn) for any questions and cooperations.

