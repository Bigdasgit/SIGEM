'''
Created on May 15, 2025
The formal implementation of SIGEM; 

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as tkb
from argparse import ArgumentParser
from LINOW_EMB import LINOW_bn_EMB
import time
import numpy as np
import pickle


class CustomCallback_verbose_check():
    '''
        1- Representing custom verbose informtation
        2- Applying dynamic learning rate
        3- Applying the early stopping
    '''
    def __init__(self, model, optimizer, early_stop, patience, min_delta): 
        """
           @param early_stop: boolean flag for applying early stopping rate or not.
           @param patience: Number of epochs with no improvement after which training will be stopped.
           @param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """        
        super().__init__()
        self.optimizer = optimizer
        self.model = model        
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.best_weights = None
        self.stop_training = False
        self.best_epoch = None      
              
    def on_epoch_begin(self, epoch):
        '''
            dynamic learning rate
        '''
        cr_lr = float(tkb.get_value(self.optimizer.learning_rate))                  
        if 0 <epoch<100 and (epoch%10)==0:
            new_lr = cr_lr - (0.1*cr_lr) 
            if new_lr > 1e-6:
                self.optimizer.learning_rate.assign(new_lr)
                                       
    def on_epoch_end(self, epoch_loss, epoch):
        '''
            epoch verbose + early stopping 
        '''
        ####################################################
            # print epoch info
        ####################################################            
        print("Epoch {}, lr {}, Loss {}".format(epoch+1,round(float(tkb.get_value(self.optimizer.learning_rate)),5),round(float(epoch_loss.numpy()),4)))
        
        ####################################################
            # savig best weights + applying early stopping
        ####################################################            
        if epoch_loss < self.best_loss - self.min_delta:
            self.best_loss = epoch_loss.numpy()
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch+1
            
        elif self.early_stop: 
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                self.model.set_weights(self.best_weights)
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print("Model's weights are restored to the best ones at epoch {} with loss {}.".format(self.best_epoch,round(float(self.best_loss),4)))
                                        
    def on_train_end(self):
        '''
            train's stop verbose 
        '''
        self.model.set_weights(self.best_weights)
        print("Training is finished; model's weights are restored to the best ones in epoch {}; .... Final loss: {}".format(self.best_epoch,round(float(self.best_loss),4)))

class CustomDenseLayer(layers.Layer):
    '''
        A custom layer in shape d*|V|; 
        dot products between the latent vector of a target node and those of other nodes are calculated in 'call' function
        Weight matrix d*|V| contains the latent vectors (column i is v_i)
    '''
    def __init__(self, input_shape, output_shape, reg_rate, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.in_shape = input_shape
        self.out_shape = output_shape
        self.reg_rate = reg_rate
        
    def build(self,input_shape):
        self.kernel = self.add_weight(
            shape=(self.in_shape, self.out_shape),
            regularizer=regularizers.L2(self.reg_rate),
            name='kernel'
        )
        
    def call(self, inputs):             
        x = tf.tensordot(inputs,tf.transpose(self.kernel),axes=1) ## extract latent vectors of target nodes
        x = tf.tensordot(x, self.kernel,axes=1) ## compute dot products of latent vectors w.r.t that of target node
        return tf.nn.selu(x)

class SIGEM_model (Model):
    '''
        Single layer model
    '''
    def __init__(self, dim,num_nodes,reg_rate):
        super (SIGEM_model, self).__init__()
        self.layer_0 = CustomDenseLayer(dim, num_nodes, reg_rate, name='layer_0')          
        
    def call(self, inputs):
        return self.layer_0(inputs)
    
class ListMLELoss_topK(tf.keras.losses.Loss):
    def __init__(self, name="custom_loss"):
        super().__init__(name=name)
        
    def compute_loss_listMLE (self, y_true, y_pred):
        '''            
            @param y_true: indices of Top-k nodes 
            @param y_pred: model's output
            The actual loss is implemented by tf.function (i.e., compute_loss_tfFunc); 
            to have a better performance, we do not call tf.function directly inside the loop (i.e., train_one_epoch)
        ''' 
        return compute_loss_tfFunc(y_true, y_pred)

@tf.function   
def compute_loss_tfFunc(y_true, y_pred):
    '''            
        Computes listMLE-topK loss function
    ''' 
    raw_max = tf.reduce_max(input_tensor=y_pred, axis=1, keepdims=True)
    y_pred = y_pred - raw_max       
    sum_all = tf.reduce_sum(tf.exp(y_pred), axis=1, keepdims=True)  
    y_ture_scores = tf.gather(y_pred,y_true,axis=1,batch_dims=1) 
    cumsum_y_ture_scores = tf.cumsum(tf.exp(y_ture_scores), axis=1, reverse=False, exclusive=True)    
    final_sum = sum_all - cumsum_y_ture_scores 
    loss_values = tf.math.log(tf.math.abs(final_sum) + tf.keras.backend.epsilon()) - y_ture_scores
    negative_log_likelihood = tf.reduce_mean(tf.reduce_sum(loss_values, axis=1, keepdims=True))
    return negative_log_likelihood    

def train_one_epoch(model, loss_fn, optimizer, dataset):
    '''
        execute training step for one epoch
    '''
    epoch_loss = tf.constant(0.0)
    num_batches = tf.constant(0)
    for batch_x, batch_y in dataset:    
        with tf.GradientTape() as tape:
            y_true = model(tf.sparse.to_dense(batch_x))            
            loss = loss_fn.compute_loss_listMLE(batch_y, y_true)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_loss += loss
        num_batches += 1        
    epoch_loss /= tf.cast(num_batches, tf.float32)
    return epoch_loss

def embedding(args):
    start = time.perf_counter()        
    if not os.path.exists(args.graph):
        print('ERROR: graph is invalid ...!\n')
        exit()        
        
    if args.dataset_name== None:
        print('ERROR: please inpute the dataset name ...!\n')
        exit()        
    
    if args.lr == None:
        print('ERROR: please set the learning rate; 0.0030 for small graphs and as 0.0012 for very large graphs ...!\n')
        exit()        
        
    if args.reg == None:
        print('ERROR: please set the regularization rate; 0.001 and 0.00001 with directed and undirected graphs, respectively ...!\n')
        exit()        
        
    if args.scaling_factor == None:
        print('ERROR: please set the scaling factor; 10 for link prediction and node classification, and 2 for graph reconstruction ...!\n')
        exit()        
        
    ####################################################
        # construct info string for output
    ####################################################    
    if args.gpu:
        com_type = '_GPU_bch_'+str(args.bch_gpu)+'_'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ## disabling GPU
        com_type = '_CPU_bch_'+str(args.bch_cpu)+'_' 
        com_type = com_type+'_process_'+str(args.prl_num)+'_'
    
    print()
    print(args,'\n')           
    info = args.result_dir+args.dataset_name+'_SIGEM_IT_'+str(args.itr)+'_C_'+ str(int(args.damping_factor*10))+com_type+'lr'+str(args.lr).split('.')[1]\
    +'_dim'+str(args.dim)+'_Scl_'+str(args.scaling_factor)

    ####################################################
        # Similarity Computation
    ####################################################    
    if not args.read_topK_nodes: ## compute similarity and find the topK nodes
        print(f"{'='*50} Similarity Computation {'='*50}\n".format('Similarity Computation'))
        top_indices = LINOW_bn_EMB(graph=args.graph, dataset_name=args.dataset_name, iterations=args.itr, damping_factor=0.2, bch_size_cpu=args.bch_cpu, bch_size_gpu=args.bch_gpu, 
                                        prl_num=args.prl_num, GPU=args.gpu, scaling_factor=args.scaling_factor, loss='listMLE_topK',
                                        write_topK_nodes=args.write_topK_nodes, topK_save_path=args.topK_save_path)#save topK on disk
    else: ## load the topK nodes
        print('Loading topK nodes from a binary file ...\n')
        with open(args.topK_file,'rb') as file_:
            top_indices = pickle.load(file_)
            top_indices = tf.constant(top_indices, dtype=tf.int32)
           
    ####################################################
        # Create sparse one-hot vectors
    ####################################################
    print(f"{'='*50} Model Training {'='*50}\n".format('Similarity Computation'))    
    print('Creating sparse one-hot vectors ... ')    
    indices = [[i,i] for i in range(0,len(top_indices))]
    values = np.ones (len(top_indices),dtype='float16')
    tf_input = tf.sparse.SparseTensor(indices,values,dense_shape=[len(top_indices),len(top_indices)])
    
    ####################################################
        # Create batches for training
    ####################################################
    print('Creating batches for training ... ')
    dataset = tf.data.Dataset.from_tensor_slices((tf_input, top_indices))
    if args.gpu:
        dataset = dataset.shuffle(buffer_size=2048).batch(args.bch_gpu) 
    else:    
        dataset = dataset.shuffle(buffer_size=2048).batch(args.bch_cpu)
        
    ####################################################
        # Initializing model
    ####################################################
    model = SIGEM_model(args.dim, len(top_indices),args.reg)                
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=0.5)
    loss_fn = ListMLELoss_topK()
    model.compile(optimizer=optimizer, loss=loss_fn)
    cc_verbose_check = CustomCallback_verbose_check (model, optimizer, args.early_stop, args.wait_thr, 0)
    ####################################################
        # Training
    ####################################################
    print('Training started ...\n')
    for epoch in range(args.epc):
        cc_verbose_check.on_epoch_begin(epoch)  
        epoch_loss = train_one_epoch(model, loss_fn, optimizer, dataset)
        cc_verbose_check.on_epoch_end(epoch_loss,epoch)
        if cc_verbose_check.stop_training:
            break  
    if not cc_verbose_check.stop_training:                                  
        cc_verbose_check.on_train_end()  

    ####################################################
        # Write resuls
    ####################################################        
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)        
    stop = time.perf_counter()
    time_file = open(info+'_time.txt','w')
    time_file.write("Training time: "+str(round((stop-start)/60,3))+'\n')
    time_file.close()
    
    emb_1 = model.get_layer('layer_0').weights[0][:].numpy().T ## |V|*d
    emb_file_W1 = open(info+'.emb','w')
    emb_file_W1.write(str(len(emb_1))+'\t'+str(args.dim)+'\n')
    for row in range(0,len(emb_1)):
        emb_file_W1.write(str(row))
        emb_val = ''
        for col in range(0, int(args.dim)):
            emb_val = emb_val + '\t' + str(emb_1[row][col])    
        emb_file_W1.write(emb_val+'\n')         
    emb_file_W1.close()
    print('The embedding result is written in the file .... \n')

def parse_args(graph='', dataset_name=None, result_dir='output/', dimension=128, iterations=5, damping_factor=0.2, scaling_factor=None,  
               gpu_on=True, bch_cpu=128, bch_gpu=128, prl_num=8,  epochs=100, learning_rate=None, reg_rate=None, early_stop=True, wait_thr=10, 
               read_topK_nodes=False, topK_file='', write_topK_nodes=False, topK_save_path=''):
    
    parser = ArgumentParser(description="Run SIGEM, a SImilarity based Graph EMbedding method.")
    parser.add_argument('--graph', type=str, default=graph, help='Input graph')       
    parser.add_argument('--dataset_name', type=str, default=dataset_name, help='Dataset name')   
    parser.add_argument('--result_dir', type=str, default=result_dir, help='Destination to save the embedding result')   
    parser.add_argument('--dim', type=int, default=dimension, help='Dimensionality of embeddings')    
    parser.add_argument('--itr', type=int, default=iterations, help='Number of iterations to compute LINOW scores')
    parser.add_argument('--damping_factor', type=float, default=damping_factor, help='Damping factor for similarity computation')
    parser.add_argument('--scaling_factor', type=int, default=scaling_factor, help='Scaling factor to select top nodes; set it as 10 for link prediction and node classification tasks, and as 2 for the graph reconstruction task')                                      
    parser.add_argument('--gpu', type=bool, default=gpu_on, help='Flag to whether run computation on GPU; default is True')        
    parser.add_argument('--bch_cpu', type=int, default=bch_cpu, help=' Batch size for computation on CPU; it is ignored if --gpu=True')
    parser.add_argument('--bch_gpu', type=int, default=bch_gpu, help=' Batch size for computation on GPU; it is ignored if --gpu=False')
    parser.add_argument('--prl_num', type=int, default=prl_num, help='Number of parallel computation on CPU')    
    parser.add_argument('--epc', type=int, default=epochs, help='Number of epochs')    
    parser.add_argument('--lr', type=float, default=learning_rate, help='Learning rate; set it as 0.0030 for small graphs and as 0.0012 for very large graphs')
    parser.add_argument('--reg', type=float, default=reg_rate, help='Regularization rate; set it as 0.001 and 0.00001 with directed and undirected graphs, respectively')
    parser.add_argument('--early_stop', type=bool, default=early_stop, help='Flag to apply early stopping')
    parser.add_argument('--wait_thr', type=int, default=wait_thr, help='Number of epochs with no loss improvement after which training will be stopped')
    parser.add_argument('--read_topK_nodes', type=bool, default=read_topK_nodes, help='Flag to read top nodes from a binary file; otherwise, they will be selected by computing similarity')
    parser.add_argument('--topK_file', type=str, default=topK_file, help='The paths to the saved top nodes')                   
    parser.add_argument('--write_topK_nodes', type=bool, default=write_topK_nodes, help='Flag to write top nodes into a binary file for later usage')
    parser.add_argument('--topK_save_path', type=str, default=topK_save_path, help='Path to write the top nodes into a binary file')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    embedding(args)
    # exit()
    
    #### Examples      
    # args = parse_args(graph='/home/masoud/backup_1/data/feature_learning/cora_citation/dataset/train_test/cora_citation_train_90.txt', 
    #             dataset_name='cora_citation_train_90', 
    #               result_dir='../result_test/', 
    #               dimension=128, 
    #               scaling_factor=10, 
    #               iterations=5,
    #               damping_factor= 0.2,
    #               epochs=100,
    #               bch_cpu=128,
    #               bch_gpu=128,
    #               gpu_on=True,
    #               prl_num = 2,
    #               learning_rate=0.0030, 
    #               reg_rate= 0.001,                                    
    #               early_stop=True,
    #               wait_thr=10,
    #               read_topK_nodes=True,
    #               topK_file='/home/masoud/backup_1/workspace/feature_learning/src/our_methods/SIGEM/result_test/TopKs/cora_citation_train_90_listMLE_topK_Scl10',
    #               write_topK_nodes=False,
    #               topK_save_path='/home/masoud/backup_1/workspace/feature_learning/src/our_methods/SIGEM/result_test/TopKs/'                  
    #               )
    # embedding(args)
