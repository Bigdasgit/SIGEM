# SIGEM: A Simple yet Effective Similarity based Graph Embedding Method

This repository provides the reference implementation of SIGEM and LINOW (matrix form, LINOW-sn, LINOW-bn/MP, and LINOW-bn/TF).

## Installation and Usage
In order to run SIGEM, the following packages are required:
```
Python       >= 3.12
tensorflow   >= 2.18.0
networkx     =3.4.2
numpy        =1.26.4
scipy        =1.31.1
scikit-learn =1.5.2
tqdm         =4.66.6
```
SIGEM can be run directly from the command line or migrated to your favorite IDE.
## Graph Format
A graph must be represented as a text file under the *edge list format* in which, each line corresponds to an edge in the graph, tab is used as the separator of the two nodes, and the node index is started from 0. 

## Running SIGEM
SIGEM has the following parameters: 
```
--graph: Input graph

--dataset_name: dataset name

--result_dir: Destination to save the embedding result, default is "output/" in the root directory

--dim: The embedding dimension, default is 128

--itr: Number of Iterations to compute LINOW, default is 5

--damping_factor: Damping factor for similarity computation, defaul is 0.2

--scaling_factor: Scaling factor to select top nodes; set it as 10 for link prediction and node classification tasks, and as 2 for the graph reconstruction task

--gpu: The flag indicating to run SIGEM on GPU, default is True

--bch_cpu: Batch size for computation on CPU; it is ignored if --gpu=True

--bch_gpu: Batch size for computation on GPU; it is ignored if --gpu=False

--prl_num: Number of parallel computation on CPU

--epc: Number of Epochs for training, default is 100

--lr: Learning rate; set it as 0.0030 for small graphs and as 0.0012 for very large graphs

--reg: egularization rate; set it as 0.001 and 0.00001 with directed and undirected graphs, respectively

--dynamic_lr: Flag to apply dynamic learning rate, default is True

--early_stop: Flag indicating to stop the training process if the loss stops improving, default is True

--wait_thr: Number of epochs with no loss improvement after which the training will be stopped, default is 10

--read_topK_nodes: Flag to read top nodes from a binary file, they will be selected by computing similarity otherwise; default is False

--topK_file: Paths to the saved top nodes

--write_topK_nodes: Flag to write top nodes into a binary file for later usage; default is False

--topK_save_path: Path to write the top nodes into a binary file

```
### Sample:
