
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
    --pr NAME                type of problem file to retrieve
    --max                    max number of examples to train/validate
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

    return amat

def target_to_adj_mat(target, max_n_vertices, num_edge_types, output_size, tie_fwd_bkwd=True):
    bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)

    amat = np.zeros((num_edge_types, max_n_vertices, output_size))
    #amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))

    for i, (src, e) in enumerate(target):
        try:
            amat[e-1, i+1, src] = 1
        except:
            set_trace()
        # amat[e-1 + bwd_edge_offset, src] = 1
    return amat

def adj_mat_to_target(adj_mat, is_probability=False):
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
            #TODO uncomment this
            print("adj matrix corrupted for node %d"%node)

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
        params.update({
                        'graph_state_dropout_keep_prob': 1.,
                        'task_sample_ratios': {},
                        'use_edge_bias': True,
                        'edge_weight_dropout_keep_prob': 1
                      })
        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        # inputs
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')
        #TODO: change this
        if self.args['--pr'] == 'identity':
            self.placeholders['initial_node_representation'] = tf.placeholder(
                tf.float32, [None, self.num_edge_types, None, self.params['hidden_size']],name='node_features')

        else:
            self.placeholders['initial_node_representation'] = tf.placeholder(
                tf.float32, [None, None, self.params['hidden_size']], name='node_features')

        self.placeholders['node_mask'] = tf.placeholder(
            tf.float32, [None, None], name='node_mask')
        self.placeholders['softmax_mask'] = tf.placeholder(
            tf.float32, [None, None], name='softmax_mask')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, (), name='num_vertices')
        self.placeholders['adjacency_matrix'] = tf.placeholder(
            tf.float32, [None, self.num_edge_types, None, None],
            name='adjacency_matrix')     # [b, e, v, v]
        self.__adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])         # [e, b, v, v]

        # weights
        self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, h_dim, h_dim]))
        if self.params['use_edge_bias']:
            self.weights['edge_biases'] = tf.Variable(np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32))
        with tf.variable_scope("gru_scope"):
            cell = tf.contrib.rnn.GRUCell(h_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 state_keep_prob=self.placeholders['graph_state_keep_prob'])
            self.weights['node_gru'] = cell

    def compute_final_node_representations(self) -> tf.Tensor:
        v = self.placeholders['num_vertices']
        e = self.num_edge_types
        h_dim = self.params['hidden_size']
        h = self.placeholders['initial_node_representation']   # ID: (b, e, v, h) else : (b, v, h)                    # [b, v, h]
        h = tf.reshape(h, [-1, h_dim])  # ID: (b * e * v, h) else : (b * v, h)

        with tf.variable_scope("gru_scope") as scope:
            for i in range(self.params['num_timesteps']):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    #'edge_weights' : [e, h, h]
                    m = tf.matmul(h, tf.nn.dropout(
                        self.weights['edge_weights'][edge_type],
                        keep_prob=self.placeholders['edge_weight_dropout_keep_prob'])) # ID: (b*e*v) else : [b*v, h]
                    del1 = tf.matmul(h, tf.nn.dropout(self.weights['edge_weights'][edge_type], keep_prob=self.placeholders['edge_weight_dropout_keep_prob'])) # [b*v, h]
                    self.ops['del1'] = self.weights['edge_weights'][edge_type]
                    if self.args['--pr'] == 'identity':
                        m = tf.reshape(m, [-1, e, v, h_dim])
                        m = tf.transpose(m, [1, 0, 2, 3]) # ID (e, b, v, h)

                    else:
                        m = tf.reshape(m, [-1, v, h_dim]) # [b, v, h]
                    if self.params['use_edge_bias']:
                        m += self.weights['edge_biases'][edge_type]                                         # [b, v, h]
                    if edge_type == 0:
                        # __adjacency_matrix[edge_type] (b, v, v)
                        acts = tf.matmul(self.__adjacency_matrix[edge_type], m) # ID (e, b, v, h)   (b, v, h)
                    else:
                        acts += tf.matmul(self.__adjacency_matrix[edge_type], m)# (b, v, h)

                acts = tf.reshape(acts, [-1, h_dim])  # ID (e * b * v, h)  (b * v, h)                                                       # [b*v, h]

                h = self.weights['node_gru'](acts, h)[1] # ID (e * b * v, h)  (b * v, h)                                            # [b*v, h]
            if self.args['--pr'] == 'identity':
                last_h = tf.reshape(h, [e, -1, v, h_dim])
                last_h = tf.transpose(last_h, [1, 0, 2, 3])
            else:
                last_h = tf.reshape(h, [-1, v, h_dim]) # (b, v, h)
        return last_h


    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [b x v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis = -1)        # [b, v, 2h]
        gate_input = tf.reshape(gate_input, [-1, 2 * self.params["hidden_size"]])                           # [b*v, 2h]
        last_h = tf.reshape(last_h, [-1, self.params["hidden_size"]])                                       # [b*v, h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)           # [b*v, 1]  [b*v, 3]

        e = self.num_edge_types
        v = self.placeholders['num_vertices']
        output_n = self.params['output_size']
        node_mask = self.placeholders['node_mask']
        softmax_mask = self.placeholders['softmax_mask']

        if self.args['--pr'] == 'molecule':
            gated_outputs = tf.reshape(gated_outputs, [-1,v])  # [b, v]
            masked_gated_outputs = gated_outputs * node_mask  # [b x v]
            output = tf.reduce_sum(masked_gated_outputs, axis=1)  # [b]
            self.output = output

        elif self.args['--pr'] == 'identity':
            #TODO redo mask
            # gated_outputs = gated_outputs + softmax_mask       # ( b, e * v * o)

            #TODO: trial check later if delete
            #tranform it for calculating softmax correctly
            gated_outputs = tf.reshape(gated_outputs, [-1, e, v, output_n])  # ( b, e, v, o)
            gated_outputs = tf.transpose(gated_outputs, [0, 2, 1, 3])  # ( b, v, e, o)
            gated_outputs = tf.reshape(gated_outputs, [-1, v, e * output_n]) # ( b, v, e * o)

            softmax = tf.nn.softmax(gated_outputs, axis =2)  # ID ( b, v, e * o)

            softmax1 = softmax
            softmax = tf.reshape(softmax, [-1, v, e, output_n])  # ID ( b, v, e, o)
            softmax = tf.transpose(softmax, [0, 2, 1, 3])  # ID ( b, e, v, o)
            softmax = tf.reshape(softmax, [-1, e * v * output_n])  # ID ( b, e * v * o)

            # TODO redo mask
            # softmax = tf.math.multiply(softmax, node_mask)
            self.output = tf.transpose(softmax)  # ID ( e * v * o,  b)
            # TODO: delete
            self.ops['last_h'] = last_h
            self.ops['gate_input'] = gate_input
            # self.ops['regression_gate'] = None
            # self.ops['regression_transform'] = None
            # self.ops['regression_gate'] = regression_gate(gate_input)
            # self.ops['regression_transform'] = regression_transform(last_h)

        else:
            gated_outputs = tf.reshape(gated_outputs, [-1, v])  # [b, v]
            #TODO change this
            node_mask = tf.reshape(node_mask, [-1,  v])
            gated_outputs = gated_outputs * node_mask                            # [b x v]
            softmax = tf.nn.softmax(gated_outputs)

            self.output = tf.transpose(softmax)   # ID ( e * v * o,  b)
            #TODO: delete
            self.ops['last_h'] = last_h
            self.ops['gate_input'] = gate_input

        return self.output

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool, bucket_sizes=None) -> Any:
        if bucket_sizes is None:
            bucket_sizes = np.array(list(range(4, 200, 2)))
        bucketed = defaultdict(list)
        x_dim = len(raw_data[0]["node_features"][0])
        for i, dd in enumerate(raw_data):
            d = dd
            if len(d['graph']) == 0:
                continue
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                                for v in [e[0], e[2]]]))
            n_active_nodes = len(d["node_features"])
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]

            bucketed[chosen_bucket_idx].append({
                'adj_mat':  graph_to_adj_mat_dir(d['graph'], chosen_bucket_size, self.num_edge_types), #self.params['tie_fwd_bkwd']),
                'init': d["node_features"] + [[0 for _ in range(x_dim)] for __ in
                                              range(chosen_bucket_size - n_active_nodes)],
                'labels':  self.get_labels_padded(
                    data_dict=dd, chosen_bucket_size=chosen_bucket_size,
                    n_active_nodes=n_active_nodes),
                'mask': self.get_mask(n_active_nodes=n_active_nodes, chosen_bucket_size=chosen_bucket_size),
                'softmax_mask' : self.get_mask_sm(n_active_nodes=n_active_nodes, chosen_bucket_size=chosen_bucket_size),
                'raw_sentence': d['raw_sentence'] if 'raw_sentence' in d else None
            })

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                np.random.shuffle(bucket)
                for task_id in self.params['task_ids']:
                    task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                    if task_sample_ratio is not None:
                        ex_to_sample = int(len(bucket) * task_sample_ratio)
                        for ex_id in range(ex_to_sample, len(bucket)):
                            bucket[ex_id]['labels'][task_id] = None

        bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'])]
                          for bucket_idx, bucket_data in bucketed.items()]
        bucket_at_step = [x for y in bucket_at_step for x in y]

        return (bucketed, bucket_sizes, bucket_at_step)

    def get_mask(self, n_active_nodes, chosen_bucket_size):
        if self.args['--pr'] == 'identity':
            mask = np.ones((self.num_edge_types, n_active_nodes))
            if chosen_bucket_size - n_active_nodes>0:
                #TODO fix this to output in the format of gated_outputs
                mask_zero = np.zeros((self.num_edge_types, chosen_bucket_size - n_active_nodes))
                mask = np.concatenate([mask, mask_zero], axis =-1)

            final_mask = np.reshape(mask, [-1,1])
            final_mask = np.tile(final_mask, [1, n_active_nodes])

            output_zeros = np.zeros((final_mask.shape[0], self.params['output_size'] - n_active_nodes))
            final_mask = np.concatenate([final_mask, output_zeros], axis=-1)

            return final_mask
        else:
            return [1. for _ in range(n_active_nodes) ] + [0. for _ in range(chosen_bucket_size - n_active_nodes)]

    def get_mask_sm(self, n_active_nodes, chosen_bucket_size):
        if self.args['--pr'] == 'identity':
            final_mask = np.zeros((self.num_edge_types * chosen_bucket_size, n_active_nodes))
            # output_ones = np.ones(
            #     (final_mask.shape[0], self.params['output_size'] - n_active_nodes))*-np.inf
            output_ones = np.ones(
                (final_mask.shape[0], self.params['output_size'] - n_active_nodes)) * (-100000)
            final_mask = np.concatenate([final_mask, output_ones], axis=-1)  #(e * v, o)

            return final_mask
        else:
            return [1. for _ in range(n_active_nodes) ] + [0. for _ in range(chosen_bucket_size - n_active_nodes)]

    def get_labels_padded(self, data_dict, chosen_bucket_size, n_active_nodes):
        if self.args['--pr'] == 'identity':
            return target_to_adj_mat(
                target=data_dict["targets"], max_n_vertices=chosen_bucket_size,
                num_edge_types=self.num_edge_types, output_size=self.params['output_size'],
                tie_fwd_bkwd=self.params['tie_fwd_bkwd'])
            # return [data_dict["targets"][task_id] for task_id in self.params['task_ids']]
        elif self.args['--pr'] == 'molecule':
            return [data_dict["targets"][task_id][0] for task_id in self.params['task_ids']]
        else:
            return [data_dict["targets"][task_id] + [0 for _ in range(chosen_bucket_size - n_active_nodes)] for task_id in self.params['task_ids']]
    def pad_annotations(self, annotations, chosen_bucket_size, adj_mat=None):
        if  self.args['--pr'] == 'identity':
            return self.annotations_padded_and_expanded(
                annotations=annotations, max_n_vertices=chosen_bucket_size,
                num_edge_types=self.num_edge_types, adj_mat=adj_mat)
            # return np.pad(annotations, pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.annotation_size]], mode='constant')
        else:
            return np.pad(annotations,
                           pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.annotation_size]],
                           mode='constant')

    def annotations_padded_and_expanded(self, annotations, max_n_vertices, num_edge_types, adj_mat):

        new_annotations = []
        num_vertices = np.array(adj_mat).shape[-1]

        return np.pad(adj_mat, pad_width=[[0, 0], [0, 0], [0, 0], [0, self.params['hidden_size'] - num_vertices]])
        # TODO: delete?
        # for annotation in annotations:
        #     amat = np.zeros((num_edge_types, max_n_vertices, self.params['hidden_size']))
        #     new_annotations.append(amat)

        # return new_annotations

    def make_batch(self, elements):
        batch_data = {'adj_mat': [], 'init': [], 'labels': [], 'node_mask': [], 'softmax_mask': [], 'task_masks': []}
        for d in elements:
            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['init'].append(d['init'])
            batch_data['node_mask'].append(d['mask'])
            batch_data['softmax_mask'].append(d['softmax_mask'])

            target_task_values = []
            target_task_mask = []
            for target_val in d['labels']:
                if target_val is None:  # This is one of the examples we didn't sample...
                    target_task_values.append(0.)
                    target_task_mask.append(0.)
                else:
                    target_task_values.append(target_val)
                    target_task_mask.append(1.)
            batch_data['labels'].append(target_task_values)
            batch_data['task_masks'].append(target_task_mask)
        if self.args['--pr'] == 'identity':
            batch_data['node_mask'] = np.reshape(batch_data['node_mask'], [len(elements), -1])
            batch_data['softmax_mask'] = np.reshape(batch_data['softmax_mask'], [len(elements), -1])

        return batch_data


    def make_minibatch_iterator(self, data, is_training: bool):
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                np.random.shuffle(bucketed_data)

        bucket_counters = defaultdict(int)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.

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
                adj_mat=batch_data['adj_mat'])
            # padded_labels = self.pad_labels(labels=batch_data['labels'])

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: initial_representations,
                self.placeholders['target_values']: np.transpose(np.array(batch_data['labels'])),
                self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs,
                self.placeholders['num_vertices']: bucket_sizes[bucket],
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],
                self.placeholders['node_mask']: batch_data['node_mask'],
                self.placeholders['softmax_mask']: batch_data['softmax_mask'],
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: dropout_keep_prob
            }
            bucket_counters[bucket] += 1
            iteration = step

            yield batch_feed_dict

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
        n_example_molecules = 30
        train_file_, valid_file_ = get_train_and_validation_files(self.args)
        with open(valid_file_) as valid_file:
        #with open('molecules_valid.json', 'r') as valid_file:
            example_molecules = json.load(valid_file)[:n_example_molecules]

        example_molecules, _, _ = self.process_raw_graphs(example_molecules, 
            is_training_data=False)
        # BUG: I think the code had a bug here
        # batch_data = self.make_batch(example_molecules[0])

        for value in example_molecules.keys():
            # print(self.evaluate_one_batch(batch_data['init'], batch_data['adj_mat']))
            self.evaluate_results(example_molecules, value)

    def evaluate_results(self, example_molecules, value):
        batch_data = self.make_batch(example_molecules[value])
        if self.args['--pr'] == 'identity':
            targets_and_sentence = [(x['labels'], x['raw_sentence']) for x in
                                    example_molecules[value]]
            node_masks = batch_data['node_mask']
            softmax_mask = batch_data['softmax_mask']
        else:
            targets_and_sentence = [(x['labels'][0], x['raw_sentence']) for x in example_molecules[value]]
            node_masks = np.transpose(batch_data['node_mask'])
            softmax_mask = np.transpose(batch_data['softmax_mask'])

        results = self.evaluate_one_batch(
            batch_data['init'], batch_data['adj_mat'],
            node_masks = node_masks, softmax_mask=softmax_mask)
        results = np.transpose(results) # (b, e * v * o)
        for i, result in enumerate(results):
            target, sentence = targets_and_sentence[i]
            mask = node_masks[i]
            # print("target vs predicted: %d vs %.2f for sentence :'%s'"%(target, result, sentence[:50]))
            self.interpret_result(target, sentence, result, mask)

    def interpret_result(self, target, sentence, result, mask):
        if self.args['--pr'] == 'identity':
            e, v, o = target.shape # (e, v, o)
            result_masked = np.multiply(result, mask)
            result_reshaped = np.reshape(result_masked, [e, v, o])  # (e, v, o)
            target_graph = adj_mat_to_target(adj_mat=target)
            result_graph = adj_mat_to_target(adj_mat=result_reshaped, is_probability=True)

            print("target vs predicted: %s vs %s for sentence :'%s'" % (
                target_graph, result_graph, sentence[:50]))
        else:
            print("target vs predicted: %d vs %.0f for sentence :'%s'" % (
                target.index(1), np.argmax(result), sentence[:50]))

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

            if self.args['--pr'] == 'identity':
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
            elif self.args['--pr'] == 'identity':
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
                if self.args['--pr'] == 'identity':
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

        if self.args['--pr'] == 'identity':
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

        elif self.args['--pr'] == 'identity':
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
                initial_representations, chosen_bucket_size=bucket_sizes[bucket])

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

    # try:
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
    # except:
    #     typ, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
