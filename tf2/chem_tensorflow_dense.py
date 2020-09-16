
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
    --only_labels            only labels considered
    --train_with_dev         use dev set which is smaller for training
    --input_tree_bank NAME   tree bank to use as input either std or nivre
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
import csv


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

        # add previous word edges
        prev_edge = num_edge_types - 1
        amat[prev_edge, dest, dest - 1] = 1
        # add next word edges
        prev_edge_o = prev_edge + num_edge_types
        amat[prev_edge_o, dest - 1, dest] = 1

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
        if self.args['--pr'] not in  ['btb']:
            if self.args['--pr'] in ['identity']:
                self.placeholders['initial_node_representation'] = tf.compat.v1.placeholder(
                    tf.float32, [self.num_edge_types, None, None, self.params['hidden_size']],name='node_features')
            else:
                self.placeholders['initial_node_representation'] = tf.compat.v1.placeholder(
                    tf.float32, [None, None, self.params['hidden_size']], name='node_features')

        self.placeholders['node_mask'] = tf.compat.v1.placeholder(
            tf.float32, [None, None], name='node_mask')
        self.placeholders['node_mask_edges'] = tf.compat.v1.placeholder(
            tf.float32, [None, None], name='node_mask_edges')
        self.placeholders['softmax_mask'] = tf.compat.v1.placeholder(
            tf.float32, [None, None], name='softmax_mask')
        self.placeholders['num_vertices'] = tf.compat.v1.placeholder(tf.int32, (), name='num_vertices')
        self.placeholders['sentences_id'] = tf.compat.v1.placeholder(tf.string, [None], name='sentences_id')
        self.placeholders['word_inputs']  = tf.compat.v1.placeholder(
            tf.int32, [None, None, 6], name='word_inputs')
        # [b, v, 6]
        self.placeholders['target_pos'] = tf.compat.v1.placeholder(
            tf.int32, [None, None], name='target_pos')
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
        self.weights['edge_weights'] = tf.Variable(
            glorot_init([2 * self.num_edge_types, h_dim, h_dim]))
        if self.params['use_edge_bias']:
            self.weights['edge_biases'] = tf.Variable(
                np.zeros([2 * self.num_edge_types, 1, h_dim]).astype(np.float32))

        self.weights['edge_weights_fixed'] = tf.Variable(
            glorot_init([2 * self.num_edge_types, h_dim, h_dim]))
        if self.params['use_edge_bias']:
            self.weights['edge_biases_fixed'] = tf.Variable(
                np.zeros([2 * self.num_edge_types, 1, h_dim]).astype(np.float32))

        self.weights['att_weights'] = tf.Variable(
            glorot_init([2 * self.params['hidden_size'], 2 * self.params['hidden_size']]))

        self.weights['loc_embeddings'] = tf.compat.v1.get_variable(
            'loc_embeddings', [self.max_nodes, self.loc_embedding_size],
            dtype=tf.float32)

        self.weights['head_loc_embeddings'] = tf.compat.v1.get_variable(
            'head_loc_embeddings', [self.max_nodes, self.loc_embedding_size],
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

    def random_mask(self, prob, mask_shape, dtype=tf.float32):
        """Random mask."""

        rand = tf.random.uniform(mask_shape)
        ones = tf.ones(mask_shape, dtype=dtype)
        zeros = tf.zeros(mask_shape, dtype=dtype)
        prob = tf.ones(mask_shape) * prob
        return tf.where(rand < prob, ones, zeros)

    def dropout(self, inputs, embed_keep_prob, block_out=False):
        if block_out:
            ph = tf.unstack(inputs, axis=-1)[0]
            # ph = tf.shape(inputs)[-1]
            last_dim = len(inputs.get_shape().as_list()) - 1
            drop_mask = tf.expand_dims(self.random_mask(embed_keep_prob, tf.shape(ph)), last_dim)
            # print("mask: %s " % drop_mask)
            inputs *= drop_mask
        else:
            inputs = tf.nn.dropout(inputs, 1 - embed_keep_prob) # we use placeholders so that it can change when it's validation
        return inputs

    def get_initial_node_representation(self):
        if self.args['--pr'] in ['btb']:
            h_dim = self.params['hidden_size']
            dropout_keep_prob = self.placeholders['emb_dropout_keep_prob']
            word_inputs = self.placeholders['word_inputs']  # [b, v, 2]
            loc_inputs = tf.nn.embedding_lookup(
                self.weights['loc_embeddings'], word_inputs[:, :, 0])
            loc_inputs = self.dropout(
                loc_inputs, dropout_keep_prob) # we use placeholders so that it can change when it's validation
            # BTB: [b, v, l_em]
            pos_inputs = tf.nn.embedding_lookup(
                self.weights['pos_embeddings'], word_inputs[:, :, 1])
            pos_inputs = self.dropout(
                pos_inputs, dropout_keep_prob)
            # BTB: [b, v, p_em]
            word_index_inputs = tf.nn.embedding_lookup(
                self.weights['word_embeddings'], word_inputs[:, :, 2])
            word_index_inputs = self.dropout(
                word_index_inputs, dropout_keep_prob)
            # BTB: [b, v, w_em]
            head_loc_inputs = tf.nn.embedding_lookup(
                self.weights['loc_embeddings'], word_inputs[:, :, 3])
            head_loc_inputs = self.dropout(
                head_loc_inputs, dropout_keep_prob)
            # BTB: [b, v, l_em]
            head_pos_inputs = tf.nn.embedding_lookup(
                self.weights['pos_embeddings'], word_inputs[:, :, 4])
            head_pos_inputs = self.dropout(
                head_pos_inputs, dropout_keep_prob)
            # not used didn't seem useful
            # BTB: [b, v, p_em]
            edges_inputs = tf.nn.embedding_lookup(
                self.weights['edge_embeddings'], word_inputs[:, :, 5])
            edges_inputs = self.dropout(
                edges_inputs, dropout_keep_prob)
            # BTB: [b, v, e_em]
            word_inputs_e = tf.concat(
                [loc_inputs, pos_inputs, word_index_inputs, head_loc_inputs], 2)
            # word_inputs_e = tf.concat(
            #     [pos_inputs, word_index_inputs, head_loc_inputs], 2)
            # BTB: [b, v, l_em + p_em ...]
            word_inputs_e = tf.pad(word_inputs_e, [[0, 0], [0, 0], [0, h_dim - word_inputs_e.shape[-1]]],
                                 mode='constant')
            # BTB: [b, v, h]
            return word_inputs_e
        else:
            return self.placeholders['initial_node_representation']

    def compute_final_node_representations(self, initial_node_representations, fixed_ts=None) -> tf.Tensor:
        v = self.placeholders['num_vertices']
        e = self.num_edge_types
        b = self.placeholders['num_graphs']
        h_dim = self.params['hidden_size']
        p_em = self.pos_embedding_size
        l_em = self.loc_embedding_size
        w_em = self.word_embedding_size
        h = initial_node_representations
        # BTB: [b * v, h] ID: [e, b, v, h] else : [b, v, h]    v' main dimension
        self.ops['word_inputs'] = h
        h = tf.reshape(h, [-1, h_dim])
        # BTB: [b * v, h] ID: [e * b * v, h] else : [b * v, h]
        timesteps = self.params['num_timesteps'] if fixed_ts is None else fixed_ts

        with tf.compat.v1.variable_scope("gru_scope") as scope:
            for i in range(timesteps):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()
                acts = self.compute_timestep(h, e, v, b, h_dim, fixed_ts=fixed_ts)
                # ID [e * b * v, h] else (b * v, h)
                h = self.weights['node_gru'](acts, h)[1]
                # ID [e * b * v, h]  NL (b * v, h) (b * v, h)                                      # [b*v, h]
                self.ops['h_gru'] = tf.identity(h)
            if self.args['--pr'] in ['identity']:
                last_h = tf.reshape(h, [e, -1, v, h_dim])
            else:
                last_h = tf.reshape(h, [-1, v, h_dim]) # (b, v, h)
        return last_h

    def compute_timestep(self, h, e, v, b, h_dim, fixed_ts=None):
        if self.args['--pr'] in ['identity', 'btb'] and self.args.get('--new'):
            acts = self.compute_timestep_fast(h, e, v, b, h_dim, fixed_ts)
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

    def compute_timestep_fast(self, h, e, v, b, h_dim, fixed_ts=None):
        # h: ID: [e* b* v, h] else: [b * v, h]
        # 'edge_weights' : [e, h, h]  bd: [2e, h, h]
        if self.args['--pr'] in ['identity']:
            h = tf.reshape(h, [e, -1, h_dim]) #ID: [e, b * v, h]
        if fixed_ts is None:
            m = tf.matmul(h, tf.nn.dropout(
                self.weights['edge_weights'],
                rate=1 - self.placeholders['edge_weight_dropout_keep_prob']))
        else:
            m = tf.matmul(h, tf.nn.dropout(
                self.weights['edge_weights_fixed'],
                rate=1 - self.placeholders['edge_weight_dropout_keep_prob']))
        # [e, b * v, h]  bd: [2e, b * v, h]
        self.ops['m1'] = tf.identity(m)

        if self.params['use_edge_bias']:
            #edge_biases : [e, 1, h] bd: [2e, h, h]
            if fixed_ts is None:
                m += self.weights['edge_biases']
            else:
                m += self.weights['edge_biases_fixed']

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

    def gated_regression(self, last_h, initial_node_representations, regression_gate, regression_transform,
                         second_node_representations=None, is_edge_regr=False):
        # last_h ID [e, b, v, h] else [b, v, h]
        b = self.placeholders['num_graphs']
        e = self.num_edge_types
        v = self.placeholders['num_vertices']
        output_n = self.params['output_size'] if not is_edge_regr else self.output_size_edges
        h_dim = self.params['hidden_size']
        p_em = self.pos_embedding_size
        l_em = self.loc_embedding_size
        w_em = self.word_embedding_size
        # ID [e, b, v, h] else [b, v, h]
        hidden_mult = 2 if second_node_representations is None else 3
        gate_input = tf.concat([last_h, initial_node_representations, second_node_representations], axis = -1)
        # ID [e, b, v, 2h] else [b, v, 2h]
        gate_input = tf.reshape(gate_input, [-1, hidden_mult * self.params["hidden_size"]])
        # ID [e * b * v, 2h] else [b * v, 2h]

        # gate_input = tf.matmul(gate_input, tf.nn.tanh(self.weights['att_weights'])) #  [b * v, 2h] x [2h, 2h]
        # [b * v, 2h]

        #last_h = tf.reshape(last_h, [-1, self.params["hidden_size"]])
        # ID [e * b * v, h] else [b * v, h]
        # gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)
        gated_outputs = regression_gate(gate_input)
        # BTB [b * v, o] ID [e * b * v, o] else [b * v, 1]

        node_mask = self.placeholders['node_mask']
        # BTB: #[b, v * o] ID [b, e * v * o]
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
            # gated_outputs  [b * v, o]
            # node_mask [b, v * o]
            # node_mask_edge  [b, v * e]
            gated_outputs = tf.reshape(gated_outputs, [b, v, output_n])  # [b, v, o]
            softmax = tf.nn.softmax(gated_outputs, axis=2) # [b, v, o]
            softmax = tf.reshape(softmax, [b, v * output_n]) # [b, v * o]
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
                # BTB : [e, v', v]  ID [e, v, o]
                'mask': self.get_mask(n_active_nodes=n_active_nodes, chosen_bucket_size=chosen_bucket_size),

                # BTB : [v * o]  ID [e * v, o]
                'mask_edges': self.get_mask(n_active_nodes=n_active_nodes,
                                            chosen_bucket_size=chosen_bucket_size, is_edge=True),
                # [v * e]
                # 'softmax_mask' : self.get_mask_sm(n_active_nodes=n_active_nodes, chosen_bucket_size=chosen_bucket_size),
                'raw_sentence': d['raw_sentence'] if 'raw_sentence' in d else None,
                'id' : d['id'] if 'id' in d else None,
                'words_pos': d["node_features"],
                'words_loc': [x for x in range(n_active_nodes)],
                'words_index': d["words_index"],
                'words_head': words_head,
                'words_head_pos': [d["node_features"][x] for x in words_head],
                'edges_index': edges_index,
                'target_pos' : d["node_features_target"]
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
                index_vector = np.pad(index_vector, pad_width=[0, self.bucket_max_nodes - v],
                                      mode='constant')

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

    def get_mask(self, n_active_nodes, chosen_bucket_size, is_edge = False):
        #TODO:test
        #n_active_nodes is the real number of vertices in the sentence
        if self.args['--pr'] in ['identity', 'btb']:
            e_o = self.output_size_edges
            v = chosen_bucket_size
            o = self.params['output_size']
            mask = np.ones((e_o, n_active_nodes))  # [e_o, v']
            if chosen_bucket_size - n_active_nodes>0:
                mask_zero = np.zeros((e_o, v - n_active_nodes))
                mask = np.concatenate([mask, mask_zero], axis =-1) # [e_o, v]
            if is_edge:
                final_mask = np.transpose(mask) # [v, e_o]
                final_mask = np.reshape(final_mask, [-1]) # [v * e_o]
                return final_mask
            final_mask = np.reshape(mask, [-1,1]) # [e_o * v, 1]
            final_mask = np.tile(final_mask, [1, n_active_nodes]) # (e_o * v, v')
            output_zeros = np.zeros((final_mask.shape[0], o - n_active_nodes))
            final_mask = np.concatenate([final_mask, output_zeros], axis=-1) # (e_o * v, o)

            if self.args['--pr'] in ['btb']:
                final_mask = np.reshape(final_mask, [e_o, v, o]) # BTB [e_o, v, o]
                final_mask = np.transpose(final_mask, [1, 0, 2]) # BTB [v, e_o, o]

                final_mask = np.mean(final_mask, axis=1)
                final_mask = np.reshape(final_mask, [-1])
            return final_mask  # BTB [v * o] ID [e * v, o]
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
                num_edge_types=self.output_size_edges, output_size=chosen_bucket_size,
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
                      'node_mask_edges': [], 'task_masks': [], 'sentences_id': [],
                      'words_pos':[], 'words_loc':[], 'words_index': [],
                      'words_head': [], 'words_head_pos': [], 'edges_index': [],
                      'target_pos': []}
        for d in elements:
            dd = d
            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['init'].append(d['init'])
            batch_data['node_mask'].append(d['mask'])
            batch_data['node_mask_edges'].append(d['mask_edges']) # [b, v * e]
            # batch_data['softmax_mask'].append(d['softmax_mask'])
            batch_data['sentences_id'].append(d['id'])
            batch_data['words_pos'].append(d['words_pos'])
            batch_data['words_loc'].append(d['words_loc'])
            batch_data['words_index'].append(d['words_index'])
            batch_data['words_head'].append(d['words_head'])
            batch_data['words_head_pos'].append(d['words_head_pos'])
            batch_data['edges_index'].append(d['edges_index'])
            batch_data['target_pos'].append(d['target_pos'])

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

            # batch_data['labels'] [b, e, v', v]
            target_values = self.get_target_values_formatted(
                labels=batch_data['labels'], no_labels=True)
            # BTB [b, v * o] ID: [o, v, e, b]
            target_values_edges = self.get_target_values_edges_formatted(labels=batch_data['labels'])
            # [b, v * e]
            loc_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['words_loc'], b=num_graphs, v=bucket_sizes[bucket])
            pos_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['words_pos'], b=num_graphs, v=bucket_sizes[bucket])
            word_id_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['words_index'], b=num_graphs, v=bucket_sizes[bucket])
            head_loc_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['words_head'], b=num_graphs, v=bucket_sizes[bucket])
            head_pos_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['words_head_pos'], b=num_graphs, v=bucket_sizes[bucket])
            edges_inputs = self.get_word_inputs_padded(
                words_pos=batch_data['edges_index'], b=num_graphs, v=bucket_sizes[bucket])
            # [b, v]
            target_pos = self.get_word_inputs_padded(
                words_pos=batch_data['target_pos'], b=num_graphs, v=bucket_sizes[bucket])
            # [b, v]

            word_inputs = np.stack((loc_inputs, pos_inputs, word_id_inputs,
                                    head_loc_inputs, head_pos_inputs, edges_inputs), axis=2)

            # [b, v, 6]
            batch_feed_dict = {
                self.placeholders['target_values_head']: target_values,
                #BTB [b, v  * o]  ID: [o, v, e, b] head [v, 1, b]
                self.placeholders['target_values_edges']: target_values_edges,
                # [b, v * e]
                self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
                #BTB [v, b] ID [v, b] else: [1, b]
                self.placeholders['num_graphs']: num_graphs,
                self.placeholders['num_vertices']: bucket_sizes[bucket],
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],
                #[b, e, v', v] bd: [b, 2e, v', v]
                self.placeholders['node_mask']: np.array(batch_data['node_mask']),
                # BTB [b, v * o] ID [b, e * v * o]
                self.placeholders['node_mask_edges']: np.array(batch_data['node_mask_edges']),
                # BTB [b, v * e]
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: dropout_keep_prob,
                self.placeholders['emb_dropout_keep_prob']: emb_dropout_keep_prob,
                self.placeholders['sentences_id']: batch_data['sentences_id'],
                self.placeholders['word_inputs']: word_inputs,
                # [b, v, 6]
                self.placeholders['target_pos']: target_pos
                # [b, v]
            }
            if self.args['--pr'] not in ['btb']:
                batch_feed_dict[self.placeholders['initial_node_representation']] = initial_representations,
                # ID: [e, b, v, h] else [b, v, h]
            bucket_counters[bucket] += 1

            avg_num += num_graphs

            yield batch_feed_dict

    def get_word_inputs_padded(self, words_pos, b, v):
        pos_inputs = np.zeros([b, v])
        for i, pos_list in enumerate(words_pos):
            pos_inputs[i][0:len(pos_list)] = pos_list

        return pos_inputs

    def get_target_values_edges_formatted(self, labels):
        o = self.output_size_edges
        b, e, v, _ = np.array(labels).shape

        new_labels = np.transpose(labels, axes=[0, 2, 1, 3])  #  [b, v', e, v]
        new_labels = np.sum(new_labels, axis=3) #  [b, v', e]

        new_labels = np.reshape(new_labels, [b, v * e])  # BTB [b, v' * e]

        return new_labels

    def get_target_values_formatted(self, labels, no_labels=True):
        if self.args['--pr'] in ['btb']:
            # labels [b, e, v', v]
            o = self.params['output_size']
            b, e, v, _ = np.array(labels).shape

            new_labels = np.transpose(labels, axes=[0, 2, 1, 3]) # BTB [b, v', e, v]
            new_labels = np.pad(new_labels,
                                pad_width=[[0, 0], [0, 0], [0, 0], [0, o - v]],
                                mode='constant')  # BTB [b, v', e, o]


            new_labels = np.sum(new_labels, axis=2) # BTB [b, v', o]
            new_labels = np.reshape(new_labels, [b, v * o])   # [b, v * o]

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
                           node_masks=None):
        num_vertices = len(initial_node_representations[0])

        if node_masks is None:
            node_masks = []
            for r in initial_node_representations:
                node_masks.append([1. for _ in r] + [0. for _ in range(num_vertices - len(r))])

        batch_feed_dict = {
            self.placeholders['num_graphs']: len(initial_node_representations),
            self.placeholders['num_vertices']: len(initial_node_representations[0]),
            self.placeholders['adjacency_matrix']: adjacency_matrices,
            self.placeholders['node_mask']: node_masks,
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

    def test_evaluation(self):
        test_loss, test_accs, test_errs, test_speed, test_steps, test_las, \
        test_uas, test_labels, test_values, test_v, test_masks, test_ids, \
        test_adm, test_labels_e, test_values_e, test_masks_e, test_uas_e = \
            self.run_epoch("Test run", self.test_data, False, 0)
        print("Running model on test file: %s\n"%self.params['test_file'])
        print("Test Attachment scores - LAS : %.2f%% - UAS : %.2f%% - UAS_e : %.2f%%" %
              (test_las * 100, test_uas * 100, test_uas_e * 100))
        print("%.2f\t%.2f\t%.2f" % (test_las * 100, test_uas * 100, test_uas_e * 100))

    def evaluate_results(self, data_for_batch):

        batch_data = self.make_batch(data_for_batch)
        if self.args['--pr'] in ['identity', 'btb']:
            targets_and_sentence = [(x['labels'], x['raw_sentence']) for x in
                                    data_for_batch]
            node_masks = batch_data['node_mask']

        else:
            targets_and_sentence = [(x['labels'][0], x['raw_sentence']) for x in data_for_batch]
            node_masks = np.transpose(batch_data['node_mask'])


        results = self.evaluate_one_batch(
            batch_data['init'], batch_data['adj_mat'],
            node_masks = node_masks)

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

    def get_results_reshaped(self, targets, computed_values, mask, num_vertices, is_edge=False):
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
            e, v = self.output_size_edges, num_vertices
            # e_ = 1 if self.args.get('--no_labels') else e
            o = self.params['output_size']

            if is_edge:
                # mask [b, v * e]
                # targets [b, v * e]
                results_masked = np.multiply(computed_values, mask) # [b, v * e]
                results_reshaped = np.reshape(results_masked, [-1, v, e, 1])  # [b, v, e, 1]
                results_reshaped = np.transpose(results_reshaped, [0, 2, 1, 3]) # [b, e, v, 1]

                targets_reshaped = np.reshape(targets, [-1, v, e, 1])  # [b, v, e, 1]
                targets_reshaped = np.transpose(targets_reshaped, [0, 2, 1, 3])  # [b, e, v, 1]

                mask_reshaped = np.reshape(mask, [-1, v, e, 1])  # [b, v, e, 1]
                mask_reshaped = np.transpose(mask_reshaped, [0, 2, 1, 3])  # [b, e, v, 1]
            else:
                # computed_values [b, v * e * o_]
                # mask [b, v * e * o_]
                results_masked = np.multiply(computed_values, mask) # [b, v * 1 * o_]
                results_reshaped = np.reshape(results_masked, [-1, 1 , v, o]) # [b, 1, v', o]

                targets_reshaped = np.reshape(targets, [-1, 1 , v, o]) # [b, 1, v', o]
                mask_reshaped = np.reshape(mask, [-1, 1 , v, o])

            return mask_reshaped, results_reshaped, targets_reshaped



    def humanize_all_results(
            self, all_labels, all_computed_values, all_num_vertices, all_masks, all_ids=None,
            all_adms=None, all_labels_e=None, all_computed_values_e=None, all_mask_edges=None, out_file=None):
        max_i = 50
        acc_las, acc_uas, acc_uas_e = 0, 0, 0
        processed_graphs = 0
        for i, computed_values in enumerate(all_computed_values):
            num_graphs = computed_values.shape[0]
            if i == max_i and out_file:
                break
            computed_values_e = all_computed_values_e[i] # [b, v * e]
            labels = all_labels[i]
            labels_e = all_labels_e[i]
            num_vertices = all_num_vertices[i]
            mask = all_masks[i]
            mask_edges = all_mask_edges[i]
            ids = all_ids[i]
            adms = all_adms[i]  # btb : [b, e, v, v]

            las, uas, uas_e = self.humanize_batch_results(
                labels=labels, computed_values=computed_values, num_vertices=num_vertices,
                mask=mask, ids=ids, adms=adms, labels_e=labels_e, computed_values_e=computed_values_e,
                mask_edges=mask_edges, out_file=out_file)

            acc_las   += las * num_graphs
            acc_uas   += uas * num_graphs
            acc_uas_e += uas_e * num_graphs

            processed_graphs += num_graphs

        acc_las = acc_las / processed_graphs
        acc_uas = acc_uas / processed_graphs
        acc_uas_e = acc_uas_e / processed_graphs

        return acc_las, acc_uas, acc_uas_e

    def humanize_batch_results(self, labels, computed_values, num_vertices, mask,
                                     ids=None, adms=None, labels_e=None, computed_values_e=None,
                                     mask_edges=None, word_inputs=None, target_pos=None, out_file=None):
        if self.args['--pr'] in ['btb']:
            acc_las, acc_uas, acc_uas_e = self.humanize_batch_results_btb(
                labels=labels, computed_values=computed_values, num_vertices=num_vertices,
                mask=mask, ids=ids, adms=adms, labels_e=labels_e, computed_values_e=computed_values_e,
                mask_edges=mask_edges, word_inputs=word_inputs, target_pos=target_pos, out_file=out_file)
        else:
            acc_las, acc_uas, acc_uas_e = 0, 0, 0
            self.humanize_batch_results_others(
                labels=labels, computed_values=computed_values, num_vertices=num_vertices,
                mask=mask, ids=ids, out_file=out_file)

        return acc_las, acc_uas, acc_uas_e

    def humanize_batch_results_others(self, labels, computed_values, num_vertices,
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

    def humanize_batch_results_btb(self, labels, computed_values, num_vertices,
                                   mask, ids=None, adms=None, labels_e=None,
                                   computed_values_e=None, mask_edges=None,
                                   word_inputs=None, target_pos=None, out_file=None):

        _, results_reshaped, targets_reshaped = self.get_results_reshaped(
            targets=labels, computed_values=computed_values, mask=mask,
            num_vertices=num_vertices)
        # [b, 1, v', o]

        _, results_reshaped_e, targets_reshaped_e = self.get_results_reshaped(
            targets=labels_e, computed_values=computed_values_e, mask=mask_edges,
            num_vertices=num_vertices, is_edge=True)
        # [b, e, v, 1]
        acc_las = 0
        acc_uas = 0
        acc_uas_e = 0
        b = targets_reshaped.shape[0]
        e = self.num_edge_types
        for i, result in enumerate(results_reshaped):
            result_e = results_reshaped_e[i]
            target = targets_reshaped[i]
            target_e = targets_reshaped_e[i]
            id = ids[i]
            adm = adms[i][:e, :, :]
            target_graph_h = adj_mat_to_target(adj_mat=target)
            target_graph_e = adj_mat_to_target(adj_mat=target_e)
            target_graph = self.merge_head_and_edge_graph(target_graph_h, target_graph_e)
            result_graph_h = adj_mat_to_target(
                adj_mat=result, is_probability=True, true_target=target_graph_h)
            result_graph_e = adj_mat_to_target(
                adj_mat=result_e, is_probability=True)
            result_graph = self.merge_head_and_edge_graph(result_graph_h, result_graph_e)
            input_graph =  adj_mat_to_target(adj_mat=adm)
            las, uas = self.get_las_uas(target_graph, result_graph)
            _, uas_e = self.get_las_uas(target_graph_e, result_graph_e, is_edge=True)

            acc_las += las
            acc_uas += uas
            acc_uas_e += uas_e

            if self.params.get('is_test'):
                word_input = word_inputs[i]
                target_pos_i = target_pos[i]
                self.write_results_analysis(
                    word_input=word_input, target_pos=target_pos_i,
                    result_graph=result_graph, target_graph=target_graph,
                    input_graph=input_graph, csv_file=out_file)
            else:
                if out_file:
                    self.write_to_file(id, target_graph, result_graph, input_graph, out_file=out_file)
        acc_las = acc_las/b
        acc_uas = acc_uas/b
        acc_uas_e = acc_uas_e/b

        return acc_las, acc_uas, acc_uas_e
    def write_results_analysis(self, word_input, target_pos, result_graph, target_graph, input_graph, csv_file):
        active_nodes = len(target_graph) + 1
        sentence_word_index = [int(x) for x in word_input[:, 2]][1:]

        input_heads = [0] + [x[0] for x in input_graph]
        input_edges = [0] + [x[1] for x in input_graph]

        input_word_list = [self.index_to_word_dict[x] for x in word_input[:, 2]]
        input_pos_list = [self.pos_list[int(x)] for x in word_input[:, 1]]

        input_head_word_list = [input_word_list[int(x)] for x in input_heads]
        input_head_pos_list = [input_pos_list[int(x)] for x in input_heads]
        input_head_edges = [input_edges[int(x)] for x in input_heads]

        target_heads = [0] +[x[0] for x in target_graph]
        target_edges = [0] +[x[1] for x in target_graph]
        target_pos_list = [self.pos_list_out[int(x)] for x in target_pos]

        result_heads = [0] + [x[0] for x in result_graph]
        result_edges = [0] + [x[1] for x in result_graph]
        #TODO:
        #target POS

        correct_las = [int(x == y) for x, y in zip(result_graph, target_graph)]
        correct_uas = [int(x[0] == y[0]) for x, y in zip(result_graph, target_graph)]
        correct_label = [int(x[1] == y[1]) for x, y in zip(result_graph, target_graph)]

        input_heads = input_heads[1:]
        input_edges = input_edges[1:]
        input_word_list = input_word_list[1:]
        input_pos_list = input_pos_list[1:]

        input_head_word_list = input_head_word_list[1:]
        input_head_pos_list = input_head_pos_list[1:]
        input_head_edges = input_head_edges[1:]

        target_heads = target_heads[1:]
        target_edges = target_edges[1:]
        target_pos_list = target_pos_list[1:]


        result_heads = result_heads[1:]
        result_edges = result_edges[1:]

        input_edges_l  = [self.dep_list[x - 1] for x in input_edges]
        input_head_edges_l = [(['zero']+self.dep_list)[x] for x in input_head_edges]
        target_edges_l = [self.dep_list_out[x - 1] for x in target_edges]
        result_edges_l = [self.dep_list_out[x - 1] for x in result_edges]


        writer = csv.writer(csv_file)

        for i, r_edge in enumerate(result_graph):
            row = [i+1, input_word_list[i], correct_las[i], correct_uas[i], correct_label[i],
                   input_pos_list[i], input_edges[i], input_edges_l[i], input_heads[i],
                   input_head_word_list[i], input_head_pos_list[i], input_head_edges[i],
                   input_head_edges_l[i],
                   target_heads[i], target_pos_list[i], target_edges[i], target_edges_l[i],
                   result_heads[i], result_edges[i], result_edges_l[i],
                   active_nodes]
            writer.writerow(row)


    def merge_head_and_edge_graph(self, result_graph_l, result_graph_e):
        return [[x[0], y[1]] for x, y in zip(result_graph_l, result_graph_e)]

    def write_to_file(self, id, target_graph, result_graph, input_graph, out_file):
        str_line = ''
        for i, r_edge in enumerate(result_graph):
            t_edge = target_graph[i]
            i_edge = input_graph[i]
            r_edge_f = self.get_edge_formatted(t_edge, r_edge)
            str_edge = "[%s, %s, %s]"%(str(t_edge), r_edge_f, str(i_edge))
            if i != len(result_graph) -1:
                str_edge += ' ,'
            str_line += str_edge

        out_file.write("id %s target vs predicted vs input: \n%s\n\n" % (
            id, str_line))

    def get_edge_formatted(self, t_edge, r_edge):
        r_head, r_label = r_edge
        t_head, t_label = t_edge
        head_prefix  = '*' if r_head != t_head else ''
        label_prefix = '*' if r_label != t_label else ''

        return "[%s%d, %s%d]"%(head_prefix, r_head, label_prefix, r_label)

    @staticmethod
    def get_las_uas(target_graph, result_graph, is_edge=False):
        las = 0
        uas = 0
        total_edges = len(result_graph)

        for i, result_edge in enumerate(result_graph):
            target_edge = target_graph[i]
            uas_index = 1 if is_edge else 0
            if target_edge == result_edge:
                las += 1
                uas += 1
            elif target_edge[uas_index] == result_edge[uas_index]:
                uas += 1

        return las/total_edges, uas/total_edges

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
        model.test_evaluation()
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
