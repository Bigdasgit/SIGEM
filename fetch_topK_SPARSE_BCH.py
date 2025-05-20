'''
Created on August 15, 2024

@note Fetches topK similar nodes to each node from a similarity matrix
      provides two implementation per method: pythonic and tensor based;
      the former one is used with CPU and the latter one with GPU 
            
'''
import numpy as np
import tensorflow as tf

def get_listMLE_topK_BCH_py(result_matrix, topK):    
    '''                
        @return: top_indices: |batch|*topK matrix contians indices of topK nodes to each target node 
    '''      
    top_indices = np.zeros((result_matrix.shape[0],topK),dtype='int32')  
    for target_node in range (0,result_matrix.shape[0]):
        target_node_res_sorted = np.argsort(result_matrix[target_node,:], axis=0)[::-1][:topK] 
        top_indices[target_node] = target_node_res_sorted
    return top_indices

def get_listMLE_topK_BCH_tb(result_matrix,topK):    
    '''
        @note: the sorting is performed row by row by using tf.map_fn                
        @return: top_indices: |batch|*topK tensor contians descending order of nodes 
    '''     
    def argsort_(row):                
        sorted_indices = tf.argsort(row, axis=0, direction='DESCENDING')
        return sorted_indices [:topK]
    sorted_indices_per_row = tf.map_fn(argsort_, result_matrix, fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32)) 
    return sorted_indices_per_row

