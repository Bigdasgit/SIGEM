# SIGEM: A Simple yet Effective Similarity based Graph Embedding Method

This repository provides the reference implementation of SIGEM and LINOW (matrix form, LINOW-sn, LINOW-bn/MP, and LINOW-bn/TF).

## Installation and Usage
In order to run SIGEM, the following packages are required:
```
Python       = 3.12
tensorflow   = 2.18.0
numpy        = 1.26.4
scipy        = 1.13.1
scikit-learn = 1.5.2
tqdm         = 4.66.6
```
SIGEM can be run directly from the command line or migrated to your favorite IDE.
## Graph Format
1) A graph must be represented as a text file under the **edge list format** in which, each line corresponds to an edge in the graph, tab is used as the separator of the two nodes, and the node index is started from 0.
2) A single original link in an **undirected graph and its train/test splitted graphs** must be represented via two links in both directions.

## Running SIGEM
SIGEM has the following parameters: 
```
--graph: Input graph; default is ''

--dataset_name: dataset name; default is None

--result_dir: Destination to save the embedding result, default is "output/" in the root directory

--dim: The embedding dimension, default is 128

--itr: Number of Iterations to compute LINOW, default is 5

--damping_factor: Damping factor for similarity computation, defaul is 0.2

--scaling_factor: Scaling factor to select top nodes (defaul is None); set it as 10 for link prediction and node classification tasks, and as 2 for the graph reconstruction task

--gpu: The flag indicating to run SIGEM on GPU, default is True

--bch_cpu: Batch size for computation on CPU; default is 128. It is ignored if --gpu=True

--bch_gpu: Batch size for computation on GPU; default is 128. It is ignored if --gpu=False

--prl_num: Number of parallel computation on CPU; default is 8. It is ignored if --gpu=True

--epc: Number of Epochs for training, default is 100

--lr: Learning rate (defaul is None); set it as 0.0030 for small graphs and as 0.0012 for very large graphs

--early_stop: Flag indicating to stop the training process if the loss stops improving, default is True

--wait_thr: Number of epochs with no loss improvement after which the training will be stopped, default is 10

--read_topK_nodes: Flag to read top nodes from a binary file, they will be selected by computing similarity otherwise; default is False

--topK_file: Paths to the saved top nodes

--write_topK_nodes: Flag to write top nodes into a binary file for later usage; default is False

--topK_save_path: Path to write the top nodes into a binary file
```
### NOTE:
**It is highly recommended** to set the values of "--itr", "--damping_factor", "--scaling_factor", "--bch_cpu", "--bch_gpu", and "--lr" parameters to the ones explained in Section 5.3.1 of the paper.

### Sample:
1) Link Prediction Task

```
python SIGEM.py --graph data/Cora_train_70.txt --dataset_name Cora_train_70 --scaling_factor 10 --lr 0.0030

python SIGEM.py --graph data/Cora_train_70.txt --dataset_name Cora_train_70 --scaling_factor 10 --lr 0.0030 --write_topK_nodes True --topK_save_path output/ ## saving top nodes in a file before training

python SIGEM.py --graph data/Cora_train_70.txt --dataset_name Cora_train_70 --scaling_factor 10 --lr 0.0030 --read_topK_nodes True --topK_file output/Live_train_70_SIGEM_IT_5_C_2_listMLE_topK_Scl10 ## loading top nodes from a file for training
```

2) Node Classification Task

```
python SIGEM.py --graph data/Cora_directed_graph.txt --dataset_name Cora --scaling_factor 10 --lr 0.0030

```

3) Graph Reconstruction Task

```
python SIGEM.py --graph data/Live_undirected_graph.txt --dataset_name Live --scaling_factor 2 --lr 0.0030
```

## Citation:
> Masoud Reyhani Hamedani, Jeong-Seok Oh, Seong-Un Cho, and Sang-Wook Kim. 2025. SIGEM: A Simple yet Effective Similarity based Graph Embedding Method. In Proceedings of the 31st ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining, KDD'25, August 03-07, 2025, Toronto, ON, Canada, pages: -
