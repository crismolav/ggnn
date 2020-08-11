
#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_dense.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format)
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format)
    --log_dir NAME           log dir name
    --data_dir NAME          data dir name
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
    --evaluate               example evaluation mode using a restored model
    --experiment             experiment
    --sample                 limited data
    --restrict_data K        max number of examples to train/validate
    --skip_data K            skips first K number of data samples.
    --pr NAME                type of problem file to retrieve
    --test_with_train        test with training data
    --alpha K                learning rate
    --new                    new way of calculating
    --no_labels              no labels considered
    --train_with_dev         use dev set which is smaller for training
"""
from __future__ import print_function
from typing import Sequence, Any
from docopt import docopt
from collections import defaultdict
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import json
import time
from pdb import set_trace
from chem_tensorflow import ChemModel, get_train_and_validation_files
from utils import glorot_init, MLP2
import os
import glob, os


def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=True):
    bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))

    for i, (src, e, dest) in enumerate(graph):
        amat[e-1, dest, src] = 1
        amat[e-1 + bwd_edge_offset, src, dest] = 1

    return amat

def graph_to_adj_mat_dir(graph, max_n_vertices, num_edge_types):
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))

    for i, (src, e, dest) in enumerate(graph):
        amat[e-1, dest, src] = 1
    #[e, v', v]
    return amat

def graph_to_adj_mat_bd(graph, max_n_vertices, num_edge_types):
    amat = np.zeros((2*num_edge_types, max_n_vertices, max_n_vertices))
    for i, (src, e, dest) in enumerate(graph):
        # incoming edge
        edge_index = e - 1
        amat[edge_index, dest, src] = 1
        # outgoing edge
        new_edge = edge_index + num_edge_types
        amat[new_edge, src, dest] = 1
    #[2e, v', v]
    return amat

def get_adj_mat_with_annotations(graph, node_features,  max_n_vertices, num_edge_types, annotation_size):
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices, annotation_size))
    for i, (src, e, dest) in enumerate(graph):
        #annotations start from node 0
        index_annotation = node_features[dest]
        amat[e-1, dest, src, index_annotation] = 1
    return amat

def target_to_adj_mat(target, max_n_vertices, num_edge_types, output_size, tie_fwd_bkwd=True):
    bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)

    amat = np.zeros((num_edge_types, max_n_vertices, output_size))
    #amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))

    for i, (src, e) in enumerate(target):
        amat[e-1, i+1, src] = 1

        # amat[e-1 + bwd_edge_offset, src] = 1
    #[e, v', o]
    return amat

def adj_mat_to_target(adj_mat, is_probability=False, true_target=None):
    num_e, num_v, num_o = adj_mat.shape
    graph = []

    for node in range(1, num_v):
        adj_mat_node = adj_mat[:, node, :]
        if np.amax(adj_mat_node) == 0:
            continue
        if is_probability:
            e_, src_ = np.where(adj_mat_node == np.amax(adj_mat_node))
        else:
            e_, src_ = np.where(adj_mat_node == 1)
        if len(e_) == 0:
            continue
        if len(e_) > 1 or len(src_) >1:
            pass
            # print("adj matrix corrupted for node %d :(%d, %d, %d)"%(
            #     node, true_target[node-1][0], true_target[node-1][1], node))

        e = e_[0] + 1
        src = src_[0]
        edge = [src, e]
        graph.append(edge)


    return graph


'''
Comments provide the expected tensor shapes where helpful.

Key to symbols in comments:
---------------------------
[...]:  a tensor
; ; :   a list
b:      batch size
e:      number of edge types (4)
v:      number of vertices per graph in this batch
h:      GNN hidden size
'''
# dep_tree = False
class DenseGGNNChemModel(ChemModel):
    def __init__(self, args):
        super().__init__(args)

    # @classmethod
    def default_params(self):
        params = dict(super().default_params())
        # graph_state_dropout_keep_prob is used for feeding edge_weight_dropout_keep_prob
        params.update({
                        'graph_state_dropout_keep_prob': 0.9,
                        'task_sample_ratios': {},
                        'use_edge_bias': True,
                        'edge_weight_dropout_keep_prob': 1
                      })
        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        # inputs
        self.placeholders['graph_state_keep_prob'] = tf.compat.v1.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.compat.v1.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')
        self.placeholders['emb_dropout_keep_prob'] = tf.compat.v1.placeholder(tf.float32, [],
                                                                              name='emb_dropout_keep_prob')
        #TODO: change this
        if self.args['--pr'] in ['identity']:
            self.placeholders['initial_node_representation'] = tf.compat.v1.placeholder(
                tf.float32, [self.num_edge_types, None, None, self.params['hidden_size']],name='node_features')
        else:
            self.placeholders['initial_node_representation'] = tf.compat.v1.placeholder(
                tf.float32, [None, None, self.params['hidden_size']], name='node_features')

        self.placeholders['node_mask'] = tf.compat.v1.placeholder(
            tf.float32, [None, None], name='node_mask')
        self.placeholders['softmax_mask'] = tf.compat.v1.placeholder(
            tf.float32, [None, None], name='softmax_mask')
        self.placeholders['num_vertices'] = tf.compat.v1.placeholder(tf.int32, (), name='num_vertices')
        self.placeholders['sentences_id'] = tf.compat.v1.placeholder(tf.string, [None], name='sentences_id')
        self.placeholders['word_inputs']  = tf.compat.v1.placeholder(
            tf.int32, [None, None, 6], name='word_inputs')
        # [b, v]
        self.placeholders['adjacency_matrix'] = tf.compat.v1.placeholder(
            tf.float32, [None, 2 * self.num_edge_types, None, None], name='adjacency_matrix')
        # [b, e, v', v]
        self.__adjacency_matrix = tf.transpose(a=self.placeholders['adjacency_matrix'], perm=[1, 0, 2, 3])
        # [e, b, v', v]
        # weights
        # self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, h_dim, h_dim]))
        # if self.params['use_edge_bias']:
        #     self.weights['edge_biases'] = tf.Variable(np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32))
        #weights bi directional matrix
        self.weights['edge_weights'] = tf.Variable(glorot_init([2 * self.num_edge_types, h_dim, h_dim]))
        if self.params['use_edge_bias']:
            self.weights['edge_biases'] = tf.Variable(np.zeros([2 * self.num_edge_types, 1, h_dim]).astype(np.float32))

        self.weights['loc_embeddings'] = tf.compat.v1.get_variable(
            'loc_embeddings', [self.max_nodes, self.loc_embedding_size],
            dtype=tf.float32)
        self.weights['pos_embeddings'] = tf.compat.v1.get_variable(
            'pos_embedding', [self.pos_size, self.pos_embedding_size],
            dtype=tf.float32)
        self.weights['word_embeddings'] = tf.compat.v1.get_variable(
            'word_embedding', [self.vocab_size, self.word_embedding_size],
            dtype=tf.float32)

        #+1 because num_edge_types doesnt include arbitrary 0 edge type
        self.weights['edge_embeddings'] = tf.compat.v1.get_variable(
            'edge_embeddings', [self.num_edge_types +1 , self.edge_embedding_size],
            dtype=tf.float32)

        with tf.compat.v1.variable_scope("gru_scope"):
            cell = tf.compat.v1.nn.rnn_cell.GRUCell(h_dim)
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell,
                                                 state_keep_prob=self.placeholders['graph_state_keep_prob'])
            self.weights['node_gru'] = cell

    def get_initial_node_representation(self, h_dim, p_em, l_em, w_em):
        if self.args['--pr'] in ['btb']:
            word_inputs = self.placeholders['word_inputs']  # [b, v, 2]
            loc_inputs = tf.nn.embedding_lookup(
                self.weights['loc_embeddings'], word_inputs[:, :, 0])
            loc_inputs = tf.nn.dropout(loc_inputs, 1 - (self.placeholders['emb_dropout_keep_prob'])) # we use placeholders so that it can change when it's validation
            # BTB: [b, v, l_em]
            pos_inputs = tf.nn.embedding_lookup(
                self.weights['pos_embeddings'], word_inputs[:, :, 1])
            pos_inputs = tf.nn.dropout(pos_inputs, 1 - (self.placeholders['emb_dropout_keep_prob']))
            # BTB: [b, v, p_em]
            word_index_inputs = tf.nn.embedding_lookup(
                self.weights['word_embeddings'], word_inputs[:, :, 2])
            word_index_inputs = tf.nn.dropout(word_index_inputs, 1 - (self.placeholders['emb_dropout_keep_prob']))
            # BTB: [b, v, w_em]
            head_loc_inputs = tf.nn.embedding_lookup(
                self.weights['loc_embeddings'], word_inputs[:, :, 3])
            head_loc_inputs = tf.nn.dropout(head_loc_inputs, 1 - (self.placeholders['emb_dropout_keep_prob']))
            # BTB: [b, v, l_em]
            head_pos_inputs = tf.nn.embedding_lookup(
                self.weights['pos_embeddings'], word_inputs[:, :, 4])
            head_pos_inputs = tf.nn.dropout(head_pos_inputs, 1 - (self.placeholders['emb_dropout_keep_prob']))
            # not used didn't seem useful
            # BTB: [b, v, p_em]

            edges_inputs = tf.nn.embedding_lookup(
                self.weights['edge_embeddings'], word_inputs[:, :, 5])
            edges_inputs = tf.nn.dropout(edges_inputs, 1 - (self.placeholders['emb_dropout_keep_prob']))
            # BTB: [b, v, e_em]

            word_inputs = tf.concat([loc_inputs, pos_inputs, word_index_inputs, head_loc_inputs, head_pos_inputs], 2)
            # BTB: [b, v, l_em + p_em ...]
            word_inputs = tf.pad(word_inputs, [[0, 0], [0, 0], [0, h_dim - word_inputs.shape[-1]]])
            # BTB: [b, v, h]
            return word_inputs
        else:
            return self.placeholders['initial_node_representation']

    def compute_final_node_representations(self) -> tf.Tensor:
        v = self.placeholders['num_vertices']
        e = self.num_edge_types
        b = self.placeholders['num_graphs']
        h_dim = self.params['hidden_size']
        p_em = self.pos_embedding_size
        l_em = self.loc_embedding_size
        w_em = self.word_embedding_size
        h = self.get_initial_node_representation(h_dim, p_em, l_em, w_em)
        # BTB: [b * v, h] ID: [e, b, v, h] else : [b, v, h]    v' main dimension
        self.ops['word_inputs'] = h
        h = tf.reshape(h, [-1, h_dim])
        # BTB: [b * v, h] ID: [e * b * v, h] else : [b * v, h]

        with tf.compat.v1.variable_scope("gru_scope") as scope:
            for i in range(self.params['num_timesteps']):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()
                acts = self.compute_timestep(h, e, v, b, h_dim)
                # ID [e * b * v, h] else (b * v, h)
                h = self.weights['node_gru'](acts, h)[1]
                # ID [e * b * v, h]  NL (b * v, h) (b * v, h)                                      # [b*v, h]
                self.ops['h_gru'] = tf.identity(h)
            if self.args['--pr'] in ['identity']:
                last_h = tf.reshape(h, [e, -1, v, h_dim])
            else:
                last_h = tf.reshape(h, [-1, v, h_dim]) # (b, v, h)
        return last_h

    def compute_timestep(self, h, e, v, b, h_dim):
        if self.args['--pr'] in ['identity', 'btb'] and self.args.get('--new'):
            acts = self.compute_timestep_fast(h, e, v, b, h_dim)
        else:
            acts = self.compute_timestep_normal(h, e, v, b, h_dim)
        # ID [e, b, v, h] [b, v, h]
        return acts

    def compute_timestep_normal(self, h, e, v, b, h_dim):
        #h: ID: [e* b* v, h] else : [b * v, h]
        num_edges = self.__adjacency_matrix.shape[0]

        for edge_type in range(num_edges):
            # 'edge_weights' : [e, h, h]
            m = tf.matmul(h, tf.nn.dropout(
                self.weights['edge_weights'][edge_type],
                rate=1 - (self.placeholders[
                    'edge_weight_dropout_keep_prob'])))
            # ID: [e * b * v, h] else: [b*v, h]
            self.ops['m1'] = tf.identity(m)

            if self.args['--pr'] in ['identity']:
                m = tf.reshape(m, [-1, b, v, h_dim]) # [e, b, v, h]
            else:
                m = tf.reshape(m, [b, -1, h_dim])  # [b, v, h]
            if self.params['use_edge_bias']:
                # edge_biases [e, 1, h]
                m += self.weights['edge_biases'][edge_type]
                # ID: [e, b, v, h] else [b, v, h]
            # __adjacency_matrix[edge_type] (b, v', v)
            # m ID [e, b, v, h] else [b, v, h]
            if edge_type == 0:
                acts = tf.matmul(self.__adjacency_matrix[edge_type], m)
                # ID [e, b, v, h] else [b, v, h]
            else:
                # __adjacency_matrix[edge_type] (b, v', v)
                # m: [b, v, h]
                acts += tf.matmul(self.__adjacency_matrix[edge_type], m)  # [b, v, h]

        acts = tf.reshape(acts, [-1, h_dim])
        # ID [e * b * v, h]  [b * v, h]
        self.ops['acts'] = tf.identity(acts)
        self.ops['m'] = tf.identity(m)
        self.ops['edge_weights'] = tf.identity(self.weights['edge_weights'])
        self.ops['h'] = tf.identity(h)
        self.ops['_am'] = tf.identity(self.__adjacency_matrix)

        return acts

    def compute_timestep_fast(self, h, e, v, b, h_dim):
        # h: ID: [e* b* v, h] else: [b * v, h]
        # 'edge_weights' : [e, h, h]  bd: [2e, h, h]
        if self.args['--pr'] in ['identity']:
            h = tf.reshape(h, [e, -1, h_dim]) #ID: [e, b * v, h]
        m = tf.matmul(h, tf.nn.dropout(
            self.weights['edge_weights'],
            rate=1 - self.placeholders['edge_weight_dropout_keep_prob']))
        # [e, b * v, h]  bd: [2e, b * v, h]
        self.ops['m1'] = tf.identity(m)

        if self.params['use_edge_bias']:
            #edge_biases : [e, 1, h] bd: [2e, h, h]
            m += self.weights['edge_biases']

        m = tf.reshape(m, [-1, b, v, h_dim])  #[e, b, v, h] bd: [2e, b, v, h]
        adj_m = self.__adjacency_matrix #  [e, b, v', v] bd: [2e, b, v', v]

        if self.args['--pr'] not in ['identity']:
            m = tf.transpose(m, [1, 0, 2, 3]) #  [b, e, v, h]
            m = tf.reshape(m, [b, -1, h_dim]) #  [b, e * v, h]  bd: [b, 2e * v, h]

            #TODO try other option for v
            adj_m = tf.transpose(self.__adjacency_matrix, [1, 2, 0, 3]) # [b, v', e, v]
            adj_m = tf.reshape(adj_m, [b, v, -1]) # [b, v' , e * v] bd: [b, v', 2e * v]
            # acts = tf.math.reduce_sum(acts, axis=0)  # [b, v, h]

        # adj_m  else [b, v' , e * v] ID :[e, b, v', v]
        # m      else [b, e * v, h] ID :[e, b, v, h]
        acts = tf.matmul(adj_m, m)
        # else: [b, v, h] ID [e, b, v, h]

        acts = tf.reshape(acts, [-1, h_dim])  # ID [e * b * v, h] [b * v, h]
        self.ops['acts'] = tf.identity(acts)
        self.ops['m'] = tf.identity(m)
        self.ops['edge_weights'] = tf.identity(self.weights['edge_weights'])
        self.ops['h'] = tf.identity(h)
        self.ops['_am'] = tf.identity(self.__adjacency_matrix)
        return acts

    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h ID [e, b, v, h] else [b, v, h]
        b = self.placeholders['num_graphs']
        e = self.num_edge_types
        v = self.placeholders['num_vertices']
        output_n = self.params['output_size']
        h_dim = self.params['hidden_size']
        p_em = self.pos_embedding_size
        l_em = self.loc_embedding_size
        w_em = self.word_embedding_size

        initial_node_representation = self.get_initial_node_representation(h_dim, p_em, l_em, w_em)
        gate_input = tf.concat([initial_node_representation, initial_node_representation], axis = -1)
        # ID [e, b, v, 2h] else [b, v, 2h]
        gate_input = tf.reshape(gate_input, [-1, 2 * self.params["hidden_size"]])
        # ID [e * b * v, 2h] else [b * v, 2h]
        last_h = tf.reshape(last_h, [-1, self.params["hidden_size"]])
        # ID [e * b * v, h] else [b * v, h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)
        # BTB [b * v, e * o_] ID [e * b * v, o] else [b * v, 1]

        node_mask = self.placeholders['node_mask']
        # BTB: #[b, v * e * o_] ID [b, e * v * o]
        softmax_mask = self.placeholders['softmax_mask'] # ID [b, e * v * o]

        if self.args['--pr'] == 'molecule':
            gated_outputs = tf.reshape(gated_outputs, [-1,v])  # [b, v]
            masked_gated_outputs = gated_outputs * node_mask  # [b x v]
            output = tf.reduce_sum(input_tensor=masked_gated_outputs, axis=1)  # [b]
            self.output = output

        elif self.args['--pr'] in ['identity']:
            #TODO redo mask
            # gated_outputs = gated_outputs + softmax_mask       # ( b, e * v * o)
            #tranform it for calculating softmax correctly
            gated_outputs = tf.reshape(gated_outputs, [e, -1, v, output_n]) # ID [e, b, v, o]
            gated_outputs = tf.transpose(a=gated_outputs, perm=[1, 2, 0, 3]) # [b, v, e, o]
            gated_outputs = tf.reshape(gated_outputs, [-1, v, e * output_n]) # [b, v, e * o]

            softmax = tf.nn.softmax(gated_outputs, axis =2)  # ID [b, v, e * o]

            softmax = tf.reshape(softmax, [-1, v, e, output_n])  # ID [b, v, e, o]
            softmax = tf.transpose(a=softmax, perm=[0, 2, 1, 3])  # ID [b, e, v, o]
            softmax = tf.reshape(softmax, [-1, e * v * output_n])  # ID [b, e * v * o]

            # TODO redo mask
            # softmax = tf.math.multiply(softmax, node_mask)
            self.output = tf.transpose(a=softmax)  # ID ( e * v * o,  b)
            # self.ops['regression_gate'] = None
            # self.ops['regression_transform'] = None
            # self.ops['regression_gate'] = regression_gate(gate_input)
            # self.ops['regression_transform'] = regression_transform(last_h)
        elif self.args['--pr'] in ['btb']:
            # gated_outputs  [b * v, e * o_]
            # node_mask [b, v * e * o_]
            e_ = 1 if self.args.get('--no_labels') else self.num_edge_types
            gated_outputs = tf.reshape(gated_outputs, [b, v, e_ * output_n])  # [b, v, e * o_]
            softmax = tf.nn.softmax(gated_outputs, axis=2) # [b, v, e * o_]
            softmax = tf.reshape(softmax, [b, v * e_ * output_n]) # [b, v * e * o_]
            self.output = softmax

        else:
            gated_outputs = tf.reshape(gated_outputs, [-1, v])  # [b, v]
            #TODO change this
            node_mask = tf.reshape(node_mask, [-1,  v])
            gated_outputs = gated_outputs * node_mask                            # [b x v]
            softmax = tf.nn.softmax(gated_outputs)

            self.output = tf.transpose(a=softmax)

        return self.output

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool, bucket_sizes=None) -> Any:
        if bucket_sizes is None:
            bucket_sizes = self.get_bucket_sizes()
        bucketed = defaultdict(list)
        # x_dim = self.annotation_size
        # x_dim = len(raw_data[0]["node_features"][0])
        for i, dd in enumerate(raw_data):
            d = dd
            if len(d['graph']) == 0:
                continue
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                                for v in [e[0], e[2]]]))
            n_active_nodes = len(d["node_features"])
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
            node_features_vector = self.vectorize_node_features(
                node_features=d["node_features"], v=chosen_bucket_size , graph=d['graph'])
            x_dim = len(node_features_vector[0])
            words_head = [0] + [x[0] for x in d['graph']]
            edges_index = [0] + [x[1] for x in d['graph']]
            bucketed_dict = {
                'adj_mat':  graph_to_adj_mat_bd(d['graph'], chosen_bucket_size, self.num_edge_types),
                #[e, v', v] bd [2e, v', v]
                'init': node_features_vector + [[0 for _ in range(x_dim)] for __ in
                                              range(chosen_bucket_size - n_active_nodes)],
                'labels':  self.get_labels_padded(
                    data_dict=dd, chosen_bucket_size=chosen_bucket_size,
                    n_active_nodes=n_active_nodes),
                #BTB : [e, v', v]  ID [e, v, o]
                'mask': self.get_mask(n_active_nodes=n_active_nodes, chosen_bucket_size=chosen_bucket_size),
                # 'softmax_mask' : self.get_mask_sm(n_active_nodes=n_active_nodes, chosen_bucket_size=chosen_bucket_size),
                'raw_sentence': d['raw_sentence'] if 'raw_sentence' in d else None,
                'id' : d['id'] if 'id' in d else None,
                'words_pos': d["node_features"],
                'words_loc': [x for x in range(n_active_nodes)],
                'words_index': d["words_index"],
                'words_head': words_head,
                'words_head_pos': [d["node_features"][x] for x in words_head],
                'edges_index': edges_index
            }
            bucketed[chosen_bucket_idx].append(bucketed_dict)


        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                np.random.shuffle(bucket)
                for task_id in self.params['task_ids']:
                    task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                    if task_sample_ratio is not None:
                        ex_to_sample = int(len(bucket) * task_sample_ratio)
                        for ex_id in range(ex_to_sample, len(bucket)):
                            bucket[ex_id]['labels'][task_id] = None

        bucket_at_step = [[bucket_idx for _ in range(1 + (len(bucket_data)-1) // self.params['batch_size'])]
                          for bucket_idx, bucket_data in bucketed.items()]
        bucket_at_step = [x for y in bucket_at_step for x in y]
        # if not is_training_data:

        return (bucketed, bucket_sizes, bucket_at_step)

    def get_bucket_sizes(self):
        return np.array(list(range(4, 200, 2)))

    def vectorize_node_features(self, node_features, v, graph):
        if self.args['--pr'] in ['btb']:
            vectorized_list = []
            graph_ = [[0, 0, 0]] + graph
            # pos_input = tf.nn.embedding_lookup(self.weights['pos_embeddings'], node_features)
            # set_trace()
            for i, node_feature in enumerate(node_features):
                #First we add the index of the node in the graph
                index_vector = [0] * v
                index_vector[i] = 1
                index_vector = np.pad(index_vector, pad_width=[0, self.bucket_max_nodes - v])

                #Second we add the POS of each node

                pos_vector = self.get_pos_vector(node_feature=node_feature)
                # third we add the dep vector of each node
                node_edge = graph_[i]
                dep_vector = self.get_dep_vector(node_edge=node_edge)
                node_vector = np.hstack((index_vector, pos_vector, dep_vector )).ravel()
                vectorized_list.append(node_vector)

            return vectorized_list
        # if self.args['--pr'] in ['btb']:
        #     vectorized_list = []
        #     for i, node_feature in enumerate(node_features):
        #         node_vector = [0] * (self.annotation_size)
        #         node_vector[node_feature] = 1
        #         vectorized_list.append(node_vector)
        #
        #     return vectorized_list
        else:
            return node_features

    def get_pos_vector(self, node_feature, deactivate_pos=False):
        # word_inputs = tf.nn.embedding_lookup(self.pos_embeddings, inputs[:, :, 0])
        if deactivate_pos:
            return []
        node_vector = [0] * (self.pos_size)
        node_vector[node_feature] = 1

        return node_vector

    def get_dep_vector(self, node_edge, deactivate_pos=True):
        dep_index = node_edge[1]
        if deactivate_pos:
            return []
        #in here we are arbitrarly saying that index 0 is the dep of node zero.
        node_vector = [0] * ((self.num_edge_types)+1)
        node_vector[dep_index] = 1

        return node_vector

    def get_mask(self, n_active_nodes, chosen_bucket_size):
        #TODO:test
        #n_active_nodes is the real number of vertices in the sentence
        if self.args['--pr'] in ['identity', 'btb']:
            e = self.num_edge_types
            v = chosen_bucket_size
            o = self.params['output_size']
            mask = np.ones((e, n_active_nodes))  # (e, v')
            if chosen_bucket_size - n_active_nodes>0:
                mask_zero = np.zeros((e, v - n_active_nodes))
                mask = np.concatenate([mask, mask_zero], axis =-1) # (e, v)

            final_mask = np.reshape(mask, [-1,1]) # (e * v, 1)
            final_mask = np.tile(final_mask, [1, n_active_nodes]) # (e * v, v')
            output_zeros = np.zeros((final_mask.shape[0], o - n_active_nodes))
            final_mask = np.concatenate([final_mask, output_zeros], axis=-1) # (e * v, o)

            if self.args['--pr'] in ['btb']:
                final_mask = np.reshape(final_mask, [e, v, o]) # BTB [e, v, o]
                final_mask = np.transpose(final_mask, [1, 0, 2]) # BTB [v, e, o]
                if self.args.get('--no_labels'):
                    final_mask = np.mean(final_mask, axis=1)
                final_mask = np.reshape(final_mask, [-1])

            return final_mask  # BTB [v * e * o] ID [e * v, o]
        else:
            return [1. for _ in range(n_active_nodes) ] + [0. for _ in range(chosen_bucket_size - n_active_nodes)]

    def get_mask_sm(self, n_active_nodes, chosen_bucket_size):
        if self.args['--pr'] in ['identity', 'btb']:
            final_mask = np.zeros((self.num_edge_types * chosen_bucket_size, n_active_nodes))
            second_dimension = self.params['output_size'] if self.args['--pr'] == 'identity' \
                else chosen_bucket_size
            output_ones = np.ones(
                (final_mask.shape[0], second_dimension - n_active_nodes)) * (-100000)
            final_mask = np.concatenate([final_mask, output_ones], axis=-1)  #(e * v, o)

            return final_mask
        else:
            return [1. for _ in range(n_active_nodes) ] + [0. for _ in range(chosen_bucket_size - n_active_nodes)]

    def get_labels_padded(self, data_dict, chosen_bucket_size, n_active_nodes):
        if self.args['--pr'] in ['identity']:
            return target_to_adj_mat(
                target=data_dict["targets"], max_n_vertices=chosen_bucket_size,
                num_edge_types=self.num_edge_types, output_size=self.params['output_size'],
                tie_fwd_bkwd=self.params['tie_fwd_bkwd'])
            # [e, v, o]
            # return [data_dict["targets"][task_id] for task_id in self.params['task_ids']]
        elif self.args['--pr'] == 'btb':
            return target_to_adj_mat(
                target=data_dict["targets"], max_n_vertices=chosen_bucket_size,
                num_edge_types=self.num_edge_types, output_size=chosen_bucket_size,
                tie_fwd_bkwd=self.params['tie_fwd_bkwd'])
            # [e, v', v]
        elif self.args['--pr'] == 'molecule':
            return [data_dict["targets"][task_id][0] for task_id in self.params['task_ids']]
            # else: [1]
        else:
            #TODO: check if the -1  is right
            n_active_nodes = n_active_nodes-1
            return [data_dict["targets"][task_id] + [0 for _ in range(chosen_bucket_size - n_active_nodes)] for task_id in self.params['task_ids']]
    def pad_annotations(self, annotations, chosen_bucket_size, adj_mat=None, sentences_id=None,
                        words_pos=None):
        if  self.args['--pr'] in ['identity']:
            return self.annotations_padded_and_expanded(
                annotations=annotations, max_n_vertices=chosen_bucket_size,
                num_edge_types=self.num_edge_types, adj_mat=adj_mat, sentences_id=sentences_id)
        elif self.args['--pr'] in ['btb']:
            current_size= len(annotations[0][0])
            return np.pad(annotations,
                          pad_width=[[0, 0], [0, 0],
                                     [0, self.params['hidden_size'] - current_size]],
                          mode='constant')
        else:
            return np.pad(annotations,
                           pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.annotation_size]],
                           mode='constant')

    def annotations_padded_and_expanded(self, annotations, max_n_vertices, num_edge_types,
                                        adj_mat, sentences_id):
        b, e, v, v = np.array(adj_mat).shape
        h_dim = self.params['hidden_size']
        new_annotations = np.pad(adj_mat, pad_width=[[0, 0], [0, 0], [0, 0], [0, h_dim - v]])

        new_annotations = np.transpose(new_annotations, [1, 0, 2, 3])  # [e, b, v, h]

        return new_annotations

    def pad_adj_matrix_annotations(self, adj_mat_annotations):
        # adj_mat_annotations: [b, e, v', v, a]
        new_annotations = np.pad(adj_mat_annotations,
           pad_width=[[0, 0], [0, 0], [0, 0], [0, 0], [0, self.params['hidden_size'] - self.annotation_size]],
           mode='constant') # [b, e, v', v, h]

        new_annotations = np.transpose(new_annotations, [1, 3, 0, 2, 4]) # [e, v, b, v', h]

        return new_annotations

    def make_batch(self, elements):
        batch_data = {'adj_mat': [], 'init': [], 'labels': [], 'node_mask': [],
                      'task_masks': [], 'sentences_id': [],
                      'words_pos':[], 'words_loc':[], 'words_index': [],
                      'words_head': [], 'words_head_pos': [], 'edges_index': []}
        for d in elements:
            dd = d
            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['init'].append(d['init'])
            batch_data['node_mask'].append(d['mask'])
            # batch_data['softmax_mask'].append(d['softmax_mask'])
            batch_data['sentences_id'].append(d['id'])
            batch_data['words_pos'].append(d['words_pos'])
            batch_data['words_loc'].append(d['words_loc'])
            batch_data['words_index'].append(d['words_index'])
            batch_data['words_head'].append(d['words_head'])
            batch_data['words_head_pos'].append(d['words_head_pos'])
            batch_data['edges_index'].append(d['edges_index'])

            target_task_values = []
            target_task_mask = []
            for target_val in d['labels']: # else: [1]
                if target_val is None:  # This is one of the examples we didn't sample...
                    target_task_values.append(0.)
                    target_task_mask.append(0.)
                else:
                    target_task_values.append(target_val)
                    target_task_mask.append(1.)
            batch_data['labels'].append(target_task_values)
            batch_data['task_masks'].append(target_task_mask)

        if self.args['--pr'] in ['identity']:
            #batch_data['node_mask'] BTB [b, e * v, v] ID [b, e * v, o]
            batch_data['node_mask'] = np.reshape(
                batch_data['node_mask'], [len(elements), -1])

        # batch_data['labels'] else: [b, e, v, o]
        # batch_data['task_masks') else: [b, 1]
        return batch_data

    def get_average_batch_size(self, data):
        (bucketed, bucket_sizes, bucket_at_step) = data
        bucket_counters = defaultdict(int)
        avg_num = 0
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            batch_data = self.make_batch(elements)

            num_graphs = len(batch_data['init'])
            avg_num += num_graphs

        avg_num = avg_num / len(bucket_at_step)
        return avg_num

    def make_minibatch_iterator(self, data, is_training: bool):
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                np.random.shuffle(bucketed_data)

        bucket_counters = defaultdict(int)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        emb_dropout_keep_prob = self.params['emb_dropout_keep_prob'] if is_training else 1.
        avg_num = 0
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            batch_data = self.make_batch(elements)

            num_graphs = len(batch_data['init'])
            initial_representations = batch_data['init']

            initial_representations = self.pad_annotations(
                initial_representations, chosen_bucket_size=bucket_sizes[bucket],
                adj_mat=batch_data['adj_mat'], sentences_id=batch_data['sentences_id'],
                words_pos=batch_data['words_pos'])
            #ID: [e, b, v, h] else [b, v, h]

            target_values = self.get_target_values_formatted(labels=batch_data['labels'])
            # BTB [b, v * e * o_] ID: [o, v, e, b]
            pos_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['words_pos'], b=num_graphs, v=bucket_sizes[bucket])
            loc_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['words_loc'], b=num_graphs, v=bucket_sizes[bucket])
            word_id_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['words_index'], b=num_graphs, v=bucket_sizes[bucket])
            head_loc_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['words_head'], b=num_graphs, v=bucket_sizes[bucket])
            head_pos_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['words_head_pos'], b=num_graphs, v=bucket_sizes[bucket])
            edges_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['edges_index'], b=num_graphs, v=bucket_sizes[bucket])
            # [b, v]
            word_inputs = np.stack((loc_inputs, pos_inputs, word_id_inputs,
                                    head_loc_inputs, head_pos_inputs, edges_inputs), axis=2)
            # [b, v, 6]
            batch_feed_dict = {
                self.placeholders['initial_node_representation']: initial_representations,
                # ID: [e, b, v, h] else [b, v, h]
                self.placeholders['target_values']: target_values,
                #BTB [b, v * e * o_]  ID: [o, v, e, b] head [v, 1, b]
                self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
                #BTB [v, b] ID [v, b] else: [1, b]
                self.placeholders['num_graphs']: num_graphs,
                self.placeholders['num_vertices']: bucket_sizes[bucket],
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],
                #[b, e, v', v] bd: [b, 2e, v', v]
                self.placeholders['node_mask']: np.array(batch_data['node_mask']),
                # BTB [b, v * e * o_] ID [b, e * v * o]
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: dropout_keep_prob,
                self.placeholders['emb_dropout_keep_prob']: emb_dropout_keep_prob,
                self.placeholders['sentences_id']: batch_data['sentences_id'],
                self.placeholders['word_inputs']: word_inputs
                # [b, v, 2]
            }
            bucket_counters[bucket] += 1
            iteration = step
            avg_num += num_graphs

            yield batch_feed_dict

    def get_word_inputs_padded(self, words_pos, b, v):
        pos_inputs = np.zeros([b, v])
        for i, pos_list in enumerate(words_pos):
            pos_inputs[i][0:len(pos_list)] = pos_list

        return pos_inputs

    def get_target_values_formatted(self, labels):
        if self.args['--pr'] in ['btb']:
            # labels [b, e, v', v]
            o_ = self.params['output_size']
            b, e, v, _ = np.array(labels).shape

            new_labels = np.transpose(labels, axes=[0, 2, 1, 3]) # BTB [b, v', e, v]
            new_labels = np.pad(new_labels,
                                pad_width=[[0, 0], [0, 0], [0, 0], [0, o_ - v]],
                                mode='constant')  # BTB [b, v', e, o_]
            if self.args.get('--no_labels'):
                new_labels = np.sum(new_labels, axis=2) # BTB [b, v', o_]
                e = 1
            new_labels = np.reshape(new_labels, [b, v * e * o_])   # BTB [b, v * e * o_]
            return new_labels
        else:
            return np.transpose(np.array(labels))

    def pad_labels(self, labels):
        if isinstance(labels[0][0], int):
            return labels
        max_length = max([len(label) for label_list in labels for label in label_list])
        new_labels =  [self.pad_label_list(max_length, label_list) for label_list in labels]
        return new_labels

    def pad_label_list(self, num_zeros, label_list):
        return [self.pad_label(num_zeros, label) for label in label_list]
    def pad_label(self, num_zeros, label):
        new_label = label + [0] * (num_zeros-len(label))
        return new_label

    def evaluate_one_batch(self, initial_node_representations, adjacency_matrices,
                           node_masks=None, softmax_mask=None):
        num_vertices = len(initial_node_representations[0])

        if node_masks is None:
            node_masks = []
            for r in initial_node_representations:
                node_masks.append([1. for _ in r] + [0. for _ in range(num_vertices - len(r))])

        batch_feed_dict = {
            self.placeholders['initial_node_representation']: self.pad_annotations(
                initial_node_representations, chosen_bucket_size=num_vertices, adj_mat=np.array(adjacency_matrices)),
            self.placeholders['num_graphs']: len(initial_node_representations),
            self.placeholders['num_vertices']: len(initial_node_representations[0]),
            self.placeholders['adjacency_matrix']: adjacency_matrices,
            self.placeholders['node_mask']: node_masks,
            self.placeholders['softmax_mask']: softmax_mask,
            self.placeholders['graph_state_keep_prob']: 1.0,
            self.placeholders['out_layer_dropout_keep_prob']: 1.0,
            self.placeholders['edge_weight_dropout_keep_prob']: 1.0
        }
        #TODO change this, it was only one value before
        fetch_list = self.output

        result = self.sess.run(fetch_list, feed_dict=batch_feed_dict)

        return result

    def example_evaluation(self):
        ''' Demonstration of what test-time code would look like
        we query the model with the first n_example_molecules from the validation file
        '''
        n_example_molecules = 3
        train_file_, valid_file_ = get_train_and_validation_files(self.args)
        with open(valid_file_) as valid_file:
        #with open('molecules_valid.json', 'r') as valid_file:
            example_molecules = json.load(valid_file)[:n_example_molecules]

        example_molecules, _, _ = self.process_raw_graphs(example_molecules, 
            is_training_data=False)
        # BUG: I think the code had a bug here
        # batch_data = self.make_batch(example_molecules[0])

        acc_las, acc_uas = 0, 0
        total_examples = 0
        for value in example_molecules.keys():
            b = len(example_molecules[value])
            # print(self.evaluate_one_batch(batch_data['init'], batch_data['adj_mat']))
            las, uas = self.evaluate_results(example_molecules[value])
            total_examples += b
            acc_las += las * b
            acc_uas += uas * b

        acc_las = acc_las / total_examples
        acc_uas = acc_uas / total_examples
        print("Attachment scores - LAS : %.2f - UAS : %.2f" % (acc_las, acc_uas))

    def evaluate_results(self, data_for_batch):

        batch_data = self.make_batch(data_for_batch)
        if self.args['--pr'] in ['identity', 'btb']:
            targets_and_sentence = [(x['labels'], x['raw_sentence']) for x in
                                    data_for_batch]
            node_masks = batch_data['node_mask']
            softmax_mask = batch_data['softmax_mask']
        else:
            targets_and_sentence = [(x['labels'][0], x['raw_sentence']) for x in data_for_batch]
            node_masks = np.transpose(batch_data['node_mask'])
            softmax_mask = np.transpose(batch_data['softmax_mask'])

        results = self.evaluate_one_batch(
            batch_data['init'], batch_data['adj_mat'],
            node_masks = node_masks, softmax_mask=softmax_mask)

        results = np.transpose(results)  # (b, e * v * o)

        for i, result in enumerate(results):
            target, sentence = targets_and_sentence[i]
            mask = node_masks[i]
            # print("target vs predicted: %d vs %.2f for sentence :'%s'"%(target, result, sentence[:50]))
            self.interpret_result(target, sentence, result, mask)

        if self.args['--pr'] in ['identity', 'btb']:
            b, e, v, o = np.array(batch_data['labels']).shape
            targets = np.reshape(np.array(batch_data['labels']), [b, -1])
            targets = np.transpose(targets)
            results = np.transpose(results)

            las, uas = self.get_batch_attachment_scores(
                targets=targets, computed_values=results,
                mask=batch_data['node_mask'], num_vertices=v)
            # print("Attachment scores - LAS : %.2f - UAS : %.2f" % (las, uas))
            return las, uas


    def interpret_result(self, target, sentence, result, mask):
        if self.args['--pr'] in ['identity', 'btb']:
            e, v, o = target.shape # (e, v, o)
            result_masked = np.multiply(result, mask)
            result_reshaped = np.reshape(result_masked, [e, v, o])  # (e, v, o)
            target_graph = adj_mat_to_target(adj_mat=target)
            result_graph = adj_mat_to_target(
                adj_mat=result_reshaped, is_probability=True, true_target=target_graph)

            print("target vs predicted: \n%s vs \n%s for sentence :'%s'" % (
                target_graph, result_graph, sentence[:50]))
        else:
            print("target vs predicted: %d vs %.0f for sentence :'%s'" % (
                target.index(1), np.argmax(result), sentence[:50]))


    def get_batch_attachment_scores(self, targets, computed_values, mask,
                                    num_vertices, sentences_id, adjacency_matrix=None):
        if self.args['--pr'] not in ['identity', 'btb']:
            return 0, 0
        else:
            return self.get_batch_attachment_scores_(
                targets=targets, computed_values=computed_values,
                mask=mask, num_vertices=num_vertices, sentences_id=sentences_id,
                adjacency_matrix=adjacency_matrix)

    def get_batch_attachment_scores_(self, targets, computed_values, mask, num_vertices,
                                       sentences_id, adjacency_matrix=None):
        # mask  [b,  e * v * o]
        # computed_values  [b, v * e * o_] [e * v * o, b]
        # target = [b, v * e * o_] labels [e * v * o, b]
        _, results_reshaped, targets_reshaped = self.get_results_reshaped(
            targets=targets, computed_values=computed_values, mask=mask,
            num_vertices=num_vertices)
        #BTB [b, e, v, o_]
        acc_las = 0
        acc_uas = 0
        b = targets_reshaped.shape[0]

        for i, result in enumerate(results_reshaped):
            target = targets_reshaped[i] # [e, v, o_]
            adj_m = adjacency_matrix[i]
            input_matrix = np.sum(adj_m, axis=0)
            input_matrix = np.reshape(input_matrix, [1, input_matrix.shape[0], input_matrix.shape[1]])

            target_graph = adj_mat_to_target(adj_mat=target)
            input_graph = adj_mat_to_target(adj_mat=input_matrix)
            result_graph = adj_mat_to_target(
                adj_mat=result, is_probability=True, true_target=target_graph)
            las, uas = self.get_las_uas(target_graph, result_graph)

            acc_las += las
            acc_uas += uas

        return acc_las/b, acc_uas/b

    def get_results_reshaped(self, targets, computed_values, mask, num_vertices, adms=None):
        if self.args['--pr'] in ['identity']:
            e, v, o = self.num_edge_types, num_vertices, self.params['output_size']
            e_ = 1 if self.args.get('--no_labels') else e
            results = np.transpose(computed_values)  # (b, e * v * o)
            results_masked = np.multiply(results, mask)  # (b, e * v * o) # NL (b, v * o)
            results_reshaped = np.reshape(results_masked, [-1, e_, v, o])  # (b, e, v, o)

            targets = np.transpose(targets)  # (b, e * v * o)
            targets_reshaped = np.reshape(targets, [-1, e_, v, o])
            mask_reshaped = mask
            return mask_reshaped, results_reshaped, targets_reshaped

        elif self.args['--pr'] in ['btb']:
            e, v = self.num_edge_types, num_vertices
            e_ = 1 if self.args.get('--no_labels') else e
            o_ = self.params['output_size']

            mask_reshaped = mask  # [b, v * e * o_]
            results_masked = np.multiply(computed_values, mask_reshaped) # [b, v * e * o_]
            results_reshaped = np.reshape(results_masked, [-1, e_, v, o_]) # [b, e, v', o]

            targets_reshaped = np.reshape(targets, [-1, e_, v, o_]) # [b, e, v', o]
            # #adms [b, e, v', v]
            # adms_reshaped = None
            # if self.args.get('--no_labels') and adms is not None:
            #     adms_reshaped = np.sum(adms, axis=1) #adms [b, v', v]
            #     adms_reshaped = np.reshape(adms_reshaped, [-1, e_, v, v]) #[b, e, v', v]

            return mask_reshaped, results_reshaped, targets_reshaped

    def print_all_results_as_graph(
            self, all_labels, all_computed_values, all_num_vertices, all_masks, all_ids=None,
            all_adms=None, out_file=None):
        max_i = 8
        for i, computed_values in enumerate(all_computed_values):
            if i == max_i:
                break
            labels = all_labels[i]
            num_vertices = all_num_vertices[i]
            mask = all_masks[i]
            ids = all_ids[i]
            adms = all_adms[i]  # btb : [b, e, v, v]

            self.print_batch_results_as_graph(
                labels=labels, computed_values=computed_values, num_vertices=num_vertices,
                mask=mask, ids=ids, adms=adms, out_file=out_file)

    def print_batch_results_as_graph(self, labels, computed_values, num_vertices, mask,
                                     ids=None, adms=None, out_file=None):
        if self.args['--pr'] in ['btb']:
            self.print_batch_results_as_graph_btb(
                labels=labels, computed_values=computed_values, num_vertices=num_vertices,
                mask=mask, ids=ids, adms=adms, out_file=out_file)
        else:
            self.print_batch_results_as_graph_others(
                labels=labels, computed_values=computed_values, num_vertices=num_vertices,
                mask=mask, ids=ids, out_file=out_file)

    def print_batch_results_as_graph_others(self, labels, computed_values, num_vertices,
                                            mask, ids=None, out_file=None):
        e, v, o = self.num_edge_types, num_vertices, self.params['output_size']
        e_ = 1 if self.args.get('--no_labels') else e
        results = np.transpose(computed_values)  # (b, e * v * o)
        results_masked = np.multiply(results, mask)  # (b, e * v * o)
        results_reshaped = np.reshape(results_masked, [-1, e_, v, o])  # (b, e, v, o)

        targets = np.transpose(labels)  # (b, e * v * o)
        targets_reshaped = np.reshape(targets, [-1, e_, v, o])

        for i, result in enumerate(results_reshaped):
            target = targets_reshaped[i]
            id = ids[i]
            target_graph = adj_mat_to_target(adj_mat=target)
            result_graph = adj_mat_to_target(
                adj_mat=result, is_probability=True, true_target=target_graph)
            if out_file is None:
                print("id %s target vs predicted: \n%s vs \n%s\n" % (
                id, target_graph, result_graph))
            else:
                out_file.write("id %s target vs predicted: \n%s vs \n%s\n" % (
                id, target_graph, result_graph))

    def print_batch_results_as_graph_btb(self, labels, computed_values, num_vertices,
                                         mask, ids=None, adms=None, out_file=None):
        adms_reshaped, results_reshaped, targets_reshaped = self.get_results_reshaped(
            targets=labels, computed_values=computed_values, mask=mask,
            num_vertices=num_vertices)

        for i, result in enumerate(results_reshaped):
            target = targets_reshaped[i]
            id = ids[i]
            adm = adms[i]
            target_graph = adj_mat_to_target(adj_mat=target)
            result_graph = adj_mat_to_target(
                adj_mat=result, is_probability=True, true_target=target_graph)
            input_graph =  adj_mat_to_target(adj_mat=adm)

            if out_file is None:
                print("id %s target vs predicted: \n%s vs \n%s\n" % (
                    id, target_graph, result_graph))
            else:
                out_file.write("id %s target vs predicted vs input: \n%s vs \n\n%s vs \n\n%s\n\n" % (
                    id, target_graph, result_graph, input_graph))


    @staticmethod
    def get_las_uas(target_graph, result_graph):
        las = 0
        uas = 0
        total_edges = len(result_graph)
        for i, result_edge in enumerate(result_graph):
            target_edge = target_graph[i]
            if target_edge == result_edge:
                las += 1
                uas += 1
            elif target_edge[0] == result_edge[0]:
                uas += 1

        return las/total_edges, uas/total_edges


    def to_real_target(self, target):
        new_target = np.transpose(target)
        for edge_row in target:
            if 1 in edge_row:
                pass

    def experiment(self):
        hidden_size = self.params['hidden_size']
        is_training = True
        data = self.train_data if is_training else self.valid_data

        for step_, batch_data in enumerate(self.make_minibatch_iterator_np(data, is_training)):
            initial_representations = batch_data['initial_node_representation']
            # last_h = np.zeros_like(initial_representations)
            last_h = self.compute_final_node_representations_np(batch_data=batch_data)
            out_layer_dropout_keep_prob = 1.0

            if self.args['--pr'] in ['identity', 'btb']:
                # regression_gate = MLP2(2 * hidden_size, batch_data['num_vertices'], [], out_layer_dropout_keep_prob)
                # regression_transform = MLP2(hidden_size, batch_data['num_vertices'], [], out_layer_dropout_keep_prob)
                regression_gate = MLP2(2 * hidden_size, self.params['output_size'], [],
                                       out_layer_dropout_keep_prob)
                regression_transform = MLP2(hidden_size, self.params['output_size'], [],
                                            out_layer_dropout_keep_prob)
            else:
                regression_gate = MLP2(2 * hidden_size, 1, [], out_layer_dropout_keep_prob)
                regression_transform = MLP2(hidden_size, 1, [], out_layer_dropout_keep_prob)
            computed_values = self.gated_regression_np(
                last_h=last_h, regression_gate=regression_gate,
                regression_transform=regression_transform,
                batch_data=batch_data)
            if self.args['--pr'] == 'molecule':
                labels = batch_data['target_values'][0, :]
            elif self.args['--pr'] in ['identity', 'btb']:
                labels = batch_data['target_values']      # (o,v,e,b)
                labels = np.transpose(labels, [2,1,0,3])  # (e,v,o,b)
                labels = np.reshape(labels, [-1, batch_data['num_graphs']])  # (e*v*o,b)
            else:
                labels = batch_data['target_values'][:, 0, :]
            mask = np.transpose(batch_data['node_mask'])
            # computed_values_masked = np.ma.array(computed_values, mask=mask == 0)
            # labels_masked = np.ma.array(labels, mask=mask == 0)

            np_loss = np.sum(-np.sum(labels * np.ma.log(computed_values), axis=1))

    def compute_final_node_representations_np(self, batch_data):
        v = batch_data['num_vertices']
        h_dim = self.params['hidden_size']
        h = batch_data['initial_node_representation'] # [b, v, h]
        h = np.reshape(h, [-1, h_dim])
        edge_weights_dummy = np.zeros((self.num_edge_types, h_dim, h_dim))
        edge_biases_dummy = np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32)
        adjacency_matrix = np.transpose(batch_data['adjacency_matrix'], [1, 0, 2, 3])

        for i in range(self.params['num_timesteps']):
            for edge_type in range(self.num_edge_types):
                # m = np.matmul(h, self.weights['edge_weights'][edge_type]) # [b*v, h]
                m = np.matmul(h, edge_weights_dummy[edge_type])  # [b*v, h]
                if self.args['--pr'] in ['identity', 'btb']:
                    # m = np.reshape(m, [-1, v, self.num_edge_types, h_dim])                            # [b, v, h]
                    # m = np.transpose(m, [2, 0, 1, 3])
                    m = np.reshape(m, [self.num_edge_types, -1, v, h_dim])

                else:
                    m = np.reshape(m, [-1, v, h_dim])
                if self.params['use_edge_bias']:
                    m += edge_biases_dummy[edge_type]                                         # [b, v, h]
                if edge_type == 0:
                    acts = np.matmul(adjacency_matrix[edge_type], m)

                else:
                    acts += np.matmul(adjacency_matrix[edge_type], m)

            acts = np.reshape(acts, [-1, h_dim])                                                        # [b*v, h]
            #this is not what the original does, its a dummy
            h = acts                                                        # [b*v, h]

        if self.args['--pr'] in ['identity', 'btb']:
            last_h = np.reshape(h, [-1, self.num_edge_types, v, h_dim])
        else:
            last_h = np.reshape(h, [-1, v, h_dim])

        return last_h

    def gated_regression_np(self, last_h, regression_gate, regression_transform, batch_data):
        # last_h: [b x v x h]
        e = self.num_edge_types
        v = batch_data['num_vertices']
        output_n = self.params['output_size']

        initial_node_representation = batch_data['initial_node_representation']
        gate_input = np.concatenate([last_h, initial_node_representation], axis = -1)       # [b, v, 2h]
        gate_input = np.reshape(gate_input, [-1, 2 * self.params["hidden_size"]])                           # [b*v, 2h]
        last_h = np.reshape(last_h, [-1, self.params["hidden_size"]])                                       # [b*v, h]
        gated_outputs = self.sigmoid_array(regression_gate(gate_input)) * regression_transform(last_h)           # [b*v, o]

        output_n = self.params['output_size']
        node_mask = batch_data['node_mask']
        softmax_mask = batch_data['softmax_mask']

        if self.args['--pr'] == 'molecule':
            gated_outputs = np.reshape(gated_outputs, [-1,v])  # [b, v]
            masked_gated_outputs = gated_outputs * node_mask  # [b , v]
            output = np.sum(masked_gated_outputs, axis=1)  # [b]
            self.output = output

        elif self.args['--pr'] in ['identity', 'btb']:
            gated_outputs = np.reshape(gated_outputs, [-1, e * v * output_n])  # [b, v]
            gated_outputs = gated_outputs + softmax_mask
            #tranform it for calculating softmax correctly
            gated_outputs = np.reshape(gated_outputs, [-1, e, v, output_n])  # ( b, e, v, o)
            gated_outputs = np.transpose(gated_outputs, [0, 2, 1, 3])        # ( b, v, e, o)
            gated_outputs = np.reshape(gated_outputs, [-1, v, e * output_n]) # ( b, v, e * o)

            softmax = self.softmax(gated_outputs)  # ID ( b, v, e * o)
            softmax = np.reshape(softmax, [-1, v, e, output_n])  # ID ( b, v, e, o)
            softmax = np.transpose(softmax, [0, 2, 1, 3])  # ID ( b, e, v, o)
            softmax = np.reshape(softmax, [-1, e * v * output_n])  # ID (b, e * v * o)
            softmax = softmax * node_mask
            self.output = np.transpose(softmax)
        else:
            gated_outputs = np.reshape(gated_outputs, [-1, v])  # [b, v]
            node_mask = np.reshape(node_mask, [-1, v])
            gated_outputs = gated_outputs * node_mask                              # [b x v]
            softmax = self.softmax(gated_outputs)

            self.output = np.transpose(softmax)

        return self.output

    def make_minibatch_iterator_np(self, data, is_training: bool):
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                np.random.shuffle(bucketed_data)

        bucket_counters = defaultdict(int)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        for step in range(len(bucket_at_step)):
            # #TODO: delete

            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            batch_data = self.make_batch(elements)

            num_graphs = len(batch_data['init'])
            initial_representations = batch_data['init']

            initial_representations = self.pad_annotations(
                initial_representations, chosen_bucket_size=bucket_sizes[bucket],
                adj_mat=batch_data['adj_mat'])

            # padded_labels = self.pad_labels(labels=batch_data['labels'])
            batch_feed_dict = {
                'initial_node_representation': initial_representations,
                'target_values': np.transpose(batch_data['labels']),
                'target_mask': np.transpose(batch_data['task_masks'], axes=[1, 0]),
                'num_graphs': num_graphs,
                'num_vertices': bucket_sizes[bucket],
                'adjacency_matrix': batch_data['adj_mat'],
                'node_mask': batch_data['node_mask'],
                'softmax_mask': batch_data['softmax_mask'],
                'graph_state_keep_prob': dropout_keep_prob,
                'edge_weight_dropout_keep_prob': dropout_keep_prob
            }
            bucket_counters[bucket] += 1
            iteration = step

            yield batch_feed_dict

    def sigmoid_array(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, X):
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X))
        return expo / expo_sum


def main():
    args = docopt(__doc__)
    start_time = time.time()
    # try:
    device_name = tf.test.gpu_device_name()
    print("device_name : %s" % device_name)
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    if args['--evaluate']:
        if args['--restore'] is None:
            os.chdir(".")
            log_dir = args.get('--log_dir') or '.'
            best_models_list = sorted(glob.glob(os.path.join(log_dir,"*.pickle")), reverse=True)
            best_model = best_models_list[0]
            args['--restore'] = best_model
            print("restoring best model: %s" %best_model)

        model = DenseGGNNChemModel(args)
        model.example_evaluation()
    elif args['--experiment']:
        model = DenseGGNNChemModel(args)
        model.experiment()
    else:
        model = DenseGGNNChemModel(args)
        print("training")
        model.train()
    end_time = time.time()
    total_time = end_time - start_time
    print("total time %.2f s"%total_time)
    # except:
    #     typ, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
