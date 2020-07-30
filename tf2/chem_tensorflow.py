#!/usr/bin/env/python

import json
import os
import pickle
import random
import time
from typing import List, Any, Sequence
import numpy as np
import tensorflow as tf
from pdb import set_trace
from utils import MLP, ThreadedIterator, SMALL_NUMBER
import sys
sys.path.insert(1, './parser')
from to_graph import get_dep_and_pos_list, sample_dep_list

dep_tree = True

def get_train_and_validation_files(args):
    if args.get('--pr') == 'root':
        train_file = 'en-wsj-std-dev-stanford-3.3.0-tagged.json'
        valid_file = 'en-wsj-std-test-stanford-3.3.0-tagged.json' if not args.get('--test_with_train') else train_file
    elif args.get('--pr') == 'head':
        train_file = 'en-wsj-std-dev-stanford-3.3.0-tagged_head.json'
        valid_file = 'en-wsj-std-test-stanford-3.3.0-tagged_head.json' if not args.get('--test_with_train') else train_file

    elif args.get('--pr') == 'identity':
        if args.get('--sample'):
            train_file = 'small_dev_id.json'
            valid_file = 'small_test_id.json' if not args.get('--test_with_train') else train_file
        else:
            train_file = 'en-wsj-std-dev-stanford-3.3.0-tagged_id.json'
            valid_file = 'en-wsj-std-test-stanford-3.3.0-tagged_id.json' if not args.get('--test_with_train') else train_file

    elif args.get('--pr') == 'btb':
        train_file = 'en-wsj-std-dev-stanford-3.3.0-tagged_btb.json'
        valid_file = 'en-wsj-std-test-stanford-3.3.0-tagged_btb.json' if not args.get('--test_with_train') else train_file

    elif args.get('--pr') == 'molecule':
        train_file = 'molecules_train.json'
        valid_file = 'molecules_valid.json' if not args.get('--test_with_train') else train_file

    return train_file, valid_file

class ChemModel(object):
    # @classmethod
    def default_params(self):

        if self.args.get('--sample'):
            return self.get_id_sample_params()
        else:
            return self.get_id_params()

    def get_id_params(self):
        train_file, valid_file = get_train_and_validation_files(self.args)
        if self.args['--pr'] in ['identity', 'btb']:
            output_size = 150
        else:
            output_size = 1

        return {
            'batch_size': 30,
            'num_epochs': 500,
            'patience': 25,
            'learning_rate': 0.003 if (not self.args.get('--alpha') or self.args.get('--alpha') == '-1') else float(self.args.get('--alpha')),
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'hidden_size': 400 if self.args['--pr'] not in ['identity'] else 350,
            'num_timesteps': 4,
            'use_graph': True,

            'tie_fwd_bkwd': True,
            'task_ids': [0],

            'random_seed': 0,

            'train_file': train_file,
            'valid_file': valid_file,
            'restrict': self.args.get("--restrict_data"),
            'output_size': output_size
        }

    def get_id_sample_params(self):
        train_file, valid_file = get_train_and_validation_files(self.args)
        return {
            'batch_size': 3,
            'num_epochs': 200,
            'patience': 25,
            'learning_rate': 0.01,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'hidden_size': 10,
            'num_timesteps': 4,
            'use_graph': True,

            'tie_fwd_bkwd': True,
            'task_ids': [0],

            'random_seed': 0,

            'train_file': train_file,
            'valid_file': valid_file,
            'output_size': 1 if self.args['--pr'] not in ['identity', 'btb'] else 6
        }

    def __init__(self, args):
        self.args = args
        if 'dummy' in args:
            return
        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.valid_results_file = os.path.join(log_dir, "%s_valid.txt" % self.run_id)
        self.train_results_file = os.path.join(log_dir, "%s_train.txt" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)
        tb_log_dir = os.path.join(log_dir, "tb", self.run_id)
        os.makedirs(tb_log_dir, exist_ok=True)

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.dep_list, self.pos_list = sample_dep_list if self.args.get('--sample') else get_dep_and_pos_list(
            bank_type='std')
        self.dep_list_out, _ = sample_dep_list if self.args.get('--sample') else get_dep_and_pos_list(
            bank_type='nivre')

        self.train_data = self.load_data(params['train_file'], is_training_data=True)
        self.valid_data = self.load_data(params['valid_file'], is_training_data=False)

        # Build the actual model
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            tf.compat.v1.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()

            if not self.args.get('--experiment'):
                self.make_train_step()
                self.make_summaries()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                self.train_step_id, self.valid_step_id = self.restore_progress(restore_file)
            else:
                self.initialize_model()
                self.train_step_id = 0
                self.valid_step_id = 0

            self.train_writer = tf.compat.v1.summary.FileWriter(os.path.join(tb_log_dir, 'train'), graph=self.graph)
            self.valid_writer = tf.compat.v1.summary.FileWriter(os.path.join(tb_log_dir, 'validation'), graph=self.graph)

    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")
        skip = self.args.get("--skip_data")

        if skip is not None and int(skip) > 0:
            data = data[int(skip):]

        if restrict is not None and int(restrict) > 0:
            data = data[:int(restrict)]

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data:
            if len( g['graph']) == 0:
                continue
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))

            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))

        self.num_edge_types = max([len(self.dep_list), self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2)])

        self.annotation_size = self.get_annotation_size(data=data)

        return self.process_raw_graphs(data, is_training_data)

    def get_annotation_size(self, data):
        if self.args['--pr'] in ['btb']:
            return len(self.pos_list)
        else:
            return max(self.annotation_size, len(data[0]["node_features"][0]))

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        #TODO: refactor
        if self.args['--pr'] == 'molecule':
            self.placeholders['target_values'] = tf.compat.v1.placeholder(
                tf.float32, [len(self.params['task_ids']), None], name='target_values')
            self.placeholders['target_mask'] = tf.compat.v1.placeholder(tf.float32,
                                                              [len(self.params['task_ids']), None],
                                                              name='target_mask')
        elif self.args['--pr'] in ['identity']:
            self.placeholders['target_values'] = tf.compat.v1.placeholder(
                tf.float32, [None, None, self.num_edge_types, None], name='target_values')
            self.placeholders['target_mask'] = tf.compat.v1.placeholder(
                tf.float32, [self.num_edge_types, None], name='target_mask')
        elif self.args['--pr'] in ['btb']:
            self.placeholders['target_values'] = tf.compat.v1.placeholder(
                tf.float32, [None, None], name='target_values')
            self.placeholders['target_mask'] = tf.compat.v1.placeholder(
                tf.float32, [self.num_edge_types, None], name='target_mask')
        else:
            self.placeholders['target_values'] = tf.compat.v1.placeholder(
                tf.float32, [None, len(self.params['task_ids']), None], name='target_values')
            self.placeholders['target_mask'] = tf.compat.v1.placeholder(
                tf.float32, [len(self.params['task_ids']), None], name='target_mask')
        self.placeholders['num_graphs'] = tf.compat.v1.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.compat.v1.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')

        with tf.compat.v1.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['initial_node_representation'])

        self.ops['losses'] = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.compat.v1.variable_scope("out_layer_task%i" % task_id):
                e_ = 1 if self.args.get('--no_labels') else self.num_edge_types

                output_size =  self.params['output_size'] if self.args['--pr'] != 'btb' else self.params['output_size']*e_
                with tf.compat.v1.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], output_size, [],
                                                                           self.placeholders['out_layer_dropout_keep_prob'])
                with tf.compat.v1.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], output_size, [],
                                                                                self.placeholders['out_layer_dropout_keep_prob'])

                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id])
                # BTB [b, v * e * o_] ID [e * v * o,  b]  o is 1 for BTB
                task_target_mask = self.placeholders['target_mask'][internal_id, :]
                # ID [b] else: [b]
                task_target_num = tf.reduce_sum(input_tensor=task_target_mask) + SMALL_NUMBER
                # ID and else: b
                if self.args['--pr'] == 'molecule':
                    labels = self.placeholders['target_values'][internal_id, :]
                    mask = tf.transpose(a=self.placeholders['node_mask'])
                elif self.args['--pr'] in ['identity']:
                    labels = self.placeholders['target_values']  # [o, v, e, b]
                    labels = tf.transpose(a=labels, perm=[2, 1, 0, 3])  # [e, v, o, b]
                    labels = tf.reshape(labels, [-1, self.placeholders['num_graphs']]) # [e * v * o, b]
                    # node_mask ID [b, e * v * o]
                    mask = tf.transpose(a=self.placeholders['node_mask'])  # [e * v * o,b]
                    # ID: [e * v * o,b]
                elif self.args['--pr'] in ['btb']:
                    labels = self.placeholders['target_values']  # [b, v * e * o_]
                    mask = self.placeholders['node_mask'] #[b, v * e * o_]
                else:
                    labels = self.placeholders['target_values'][:, internal_id, :]
                    mask = tf.transpose(a=self.placeholders['node_mask'])
                # diff = computed_values - labels
                # diff = diff * task_target_mask  # Mask out unused values
                # self.ops['accuracy_task%i' % task_id] = tf.reduce_sum(tf.abs(diff)) / task_target_num
                # task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num
                # # Normalise loss to account for fewer task-specific examples in batch:
                # task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))

                # diff =  tf.math.argmax(computed_values, axis = 1) - tf.math.argmax(self.placeholders['target_values'][internal_id, :], axis = 1)
                # diff = tf.dtypes.cast(diff, tf.float32)
                #TODO: FIX THIS

                # computed_values *= task_target_mask
                # we need to redo accuracy
                # diff = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                #                                                logits=computed_values)
                # task_loss = diff
                if  self.args['--pr'] == 'molecule':
                    self.calculate_losses_for_molecules(computed_values, internal_id, task_id)
                else:
                    if self.args.get('--no_labels'):
                        computed_values, labels, mask = self.reduce_edge_dimension(
                            computed_values=computed_values, labels=labels, mask=mask)
                    new_mask = tf.cast(mask, tf.bool)
                    masked_loss = tf.boolean_mask(tensor=labels * tf.math.log(computed_values), mask= new_mask)
                    task_loss = tf.reduce_sum(input_tensor=-1*masked_loss)/task_target_num

                    #task_loss = tf.reduce_sum(-tf.reduce_sum(labels * tf.log(computed_values), axis = 1))/task_target_num
                    self.ops['accuracy_task%i' % task_id] = task_loss
                    self.ops['losses'].append(task_loss)
                    self.ops['computed_values'] = computed_values
                    self.ops['labels'] = labels
                    self.ops['node_mask'] = tf.transpose(mask)
                    self.ops['masked_loss'] = masked_loss
                    self.ops['new_mask'] =  new_mask
                    self.ops['task_target_mask'] = task_target_mask

        self.ops['loss'] = tf.reduce_sum(input_tensor=self.ops['losses'])

    def reduce_edge_dimension(self, computed_values, labels, mask):
        if self.args['--pr'] in ['btb']:
            return computed_values, labels, mask
        else:
            return self.reduce_edge_dimension_other(
                computed_values=computed_values, labels=labels, mask=mask)

    def reduce_edge_dimension_other(self, computed_values, labels, mask):
        # computed_values, labels, mask ( e * v * o,  b)

        o = self.params['output_size']
        v = self.placeholders['num_vertices']
        e = self.num_edge_types
        b = self.placeholders['num_graphs']

        computed_values = tf.reshape(computed_values, [e, v, o, b])
        labels = tf.reshape(labels, [e, v, o, b])
        mask = tf.reshape(mask, [e, v, o, b])

        computed_values = tf.math.reduce_sum(computed_values, axis = 0)
        labels = tf.math.reduce_sum(labels, axis=0)
        mask = tf.math.reduce_mean(mask, axis=0)

        computed_values = tf.reshape(computed_values, [v * o, b])
        labels = tf.reshape(labels, [v * o, b])
        mask = tf.reshape(mask, [v * o, b])

        return computed_values, labels, mask

    def reduce_edge_dimension_btb(self, computed_values, labels, mask):
        # computed_values, labels, mask [b, e * v' * v]

        o = self.params['output_size']
        v = self.placeholders['num_vertices']
        e = self.num_edge_types
        b = self.placeholders['num_graphs']

        computed_values = tf.reshape(computed_values, [b, e, v, v]) # [b, e, v', v]
        labels = tf.reshape(labels, [b, e, v, v]) # [b, e, v', v]
        mask = tf.reshape(mask,[b, e, v, v]) # [b, e, v', v]

        computed_values = tf.math.reduce_sum(computed_values, axis = 1)
        labels = tf.math.reduce_sum(labels, axis=1)
        mask = tf.math.reduce_mean(mask, axis=1)

        computed_values = tf.reshape(computed_values, [b, v * v]) # [b, v', v]
        labels = tf.reshape(labels, [b, v * v]) # [b, v', v]
        mask = tf.reshape(mask, [b, v * v]) # [b, v', v]

        return computed_values, labels, mask

    def calculate_losses_for_molecules(self, computed_values, internal_id, task_id):
        diff = computed_values - self.placeholders['target_values'][internal_id, :]
        task_target_mask = self.placeholders['target_mask'][internal_id, :]
        task_target_num = tf.reduce_sum(input_tensor=task_target_mask) + SMALL_NUMBER
        diff = diff * task_target_mask  # Mask out unused values
        self.ops['accuracy_task%i' % task_id] = tf.reduce_sum(input_tensor=tf.abs(diff)) / task_target_num
        task_loss = tf.reduce_sum(input_tensor=0.5 * tf.square(diff)) / task_target_num
        # Normalise loss to account for fewer task-specific examples in batch:
        task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
        self.ops['losses'].append(task_loss)

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.compat.v1.train.AdamOptimizer(self.params['learning_rate'],beta1=0.9, beta2=0.999)

        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)

        # Initialize newly-introduced variables:
        self.sess.run(tf.compat.v1.local_variables_initializer())

    def make_summaries(self):
        with tf.compat.v1.name_scope('summary'):
            tf.compat.v1.summary.scalar('loss', self.ops['loss'])
            for task_id in self.params['task_ids']:
                tf.compat.v1.summary.scalar('accuracy%i' % task_id, self.ops['accuracy_task%i' % task_id])
        self.ops['summary'] = tf.compat.v1.summary.merge_all()

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, epoch_name: str, data, is_training: bool, start_step: int = 0):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])

        loss = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        processed_graphs = 0
        steps = 0
        acc_las, acc_uas = 0, 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        all_labels, all_computed_values, all_num_vertices, all_masks, all_ids, all_adj_m = \
            [], [], [], [], [], []

        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            fetch_list = [self.ops['loss'], accuracy_ops, self.ops['summary'],
                          self.ops['labels'], self.ops['computed_values'],
                          self.ops['final_node_representations'],
                          self.ops['node_mask'], self.ops['losses'],
                          self.ops['masked_loss'], self.ops['new_mask'],
                          self.placeholders['initial_node_representation'],
                          self.weights['edge_weights'], self.weights['edge_biases'],
                          self.weights['regression_transform_task%i' % 0].params['weights'],
                          self.placeholders['num_vertices'], self.ops['acts'],
                          self.ops['m'], self.placeholders['adjacency_matrix'],
                          self.ops['edge_weights'], self.ops['h'], self.ops['h_gru'],
                          self.placeholders['edge_weight_dropout_keep_prob'], self.ops['m1'],
                          self.ops['_am'], self.placeholders['sentences_id'],
                          ]
            if is_training:
                #TODO: change this back to normal
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
                fetch_list.append(self.ops['train_step'])

            else:
                # it is not trainining because we are not requesting the self.ops['train_step'] parametr
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0


            result = self.sess.run(fetch_list, feed_dict=batch_data)
            #TODO: delete

            labels = result[3]
            computed_values = result[4]
            final_node_representations = result[5]
            node_mask = result[6]
            initial_node_representation = result[10]
            edge_weights = result[11]
            edge_biases = result[12]
            regression_transform_task = result[13]
            num_vertices = result[14]
            acts = result[15]
            mm = result[16]
            adjacency_matrix = result[17]
            edge_weights_ops = result[18]
            hh = result[19]
            h_gru = result[20]
            edge_weight_dkp = result[21]
            m1 = result[22]
            # _am = result[23]
            sentences_id = result[24]


            loss_ = result[0]

            # np_loss = np.sum(-np.sum(labels * np.log(computed_values), axis = 1))
            (batch_loss, batch_accuracies, batch_summary) = (result[0], result[1], result[2])
            writer = self.train_writer if is_training else self.valid_writer
            writer.add_summary(batch_summary, start_step + step)
            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)

            try:
                las, uas = self.get_batch_attachment_scores(
                    targets=labels, computed_values= computed_values,
                    mask=node_mask, num_vertices=num_vertices,
                    sentences_id=sentences_id)
                acc_las += las * num_graphs
                acc_uas += uas * num_graphs
            except:
                print('edge weights: %s'%edge_weights)
                print('edge bias: %s'%edge_biases)
                raise Exception('Apparent division by zero, comp_values: %s'%computed_values[0])

            print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
                                                                               step,
                                                                               num_graphs,
                                                                               loss / processed_graphs),
                  end='\r')
            steps += 1

            all_labels.append(labels)
            all_computed_values.append(computed_values)
            all_num_vertices.append(num_vertices)
            all_masks.append(node_mask)
            all_ids.append(sentences_id)
            all_adj_m.append(adjacency_matrix)

        # if not is_training:
        #     set_trace()
        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        instance_per_sec = processed_graphs / (time.time() - start_time)
        acc_las = acc_las / processed_graphs
        acc_uas = acc_uas / processed_graphs
        # if acc_las > 0.65:
        #     self.print_all_results_as_graph(
        #         all_labels=all_labels, all_computed_values=all_computed_values,
        #         all_num_vertices=all_num_vertices, all_masks=all_masks)

        return loss, accuracies, error_ratios, instance_per_sec, steps, acc_las, acc_uas, \
               all_labels, all_computed_values, all_num_vertices, all_masks, \
               all_ids, all_adj_m

    def train(self):
        log_to_save = []
        total_time_start = time.time()
        avg_train_batch_size = self.get_average_batch_size(data=self.train_data)
        print("Average train batch size: %.2f\n" % avg_train_batch_size)
        avg_val_batch_size   = self.get_average_batch_size(data=self.valid_data)
        print("Average val batch size: %.2f\n" % avg_val_batch_size)
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                _, valid_accs, _, _, steps, valid_las, valid_uas, _, _, _, _, _ = \
                    self.run_epoch("Resumed (validation)", self.valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (float("+inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_accs, train_errs, train_speed, train_steps, train_las,\
                train_uas, train_labels, train_values, train_v, train_masks, train_ids, train_adm = \
                    self.run_epoch("epoch %i (training)" % epoch, self.train_data, True, self.train_step_id)
                self.train_step_id += train_steps
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                print("\r\x1b[K Train: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (train_loss,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        train_speed))
                print("Train Attachment scores - LAS : %.1f%% - UAS : %.1f%%" % (train_las*100, train_uas*100))
                valid_loss, valid_accs, valid_errs, valid_speed, valid_steps, valid_las, \
                valid_uas, valid_labels, valid_values, valid_v, valid_masks, valid_ids, valid_adm = \
                    self.run_epoch("epoch %i (validation)" % epoch, self.valid_data, False, self.valid_step_id)
                self.valid_step_id += valid_steps

                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print("\r\x1b[K Valid: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (valid_loss,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        valid_speed))
                print("Valid Attachment scores - LAS : %.1f%% - UAS : %.1f%%" % (valid_las*100, valid_uas*100))
                epoch_time = time.time() - total_time_start
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': (train_loss, train_accs.tolist(), train_errs.tolist(), train_speed),
                    'valid_results': (valid_loss, valid_accs.tolist(), valid_errs.tolist(), valid_speed),
                }
                log_to_save.append(log_entry)
                with open(self.log_file, 'w') as f:
                    json.dump(log_to_save, f, indent=4)

                val_acc = train_loss
                #TODO: reconsider this change, we are now using loss as accuracy
                # val_acc = np.sum(valid_accs)  # type: float
                # val_acc = 1-valid_las
                # if val_acc < best_val_acc:

                ##here look at train_las and print valid las
                if val_acc < best_val_acc:

                    self.save_progress(self.best_model_file, self.train_step_id, self.valid_step_id)
                    print("  (Best epoch so far, cum. val. acc decreased to %.5f from %.5f. Saving to '%s')" % (
                        val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch

                    self.save_results(
                        labels=valid_labels, values=valid_values, num_vertices=valid_v,
                        masks=valid_masks, ids=valid_ids, adm=valid_adm, train=False)

                    self.save_results(
                        labels=train_labels, values=train_values, num_vertices=train_v,
                        masks=train_masks, ids=train_ids, adm=train_adm, train=True)

                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params['patience'])
                    break

    def save_results(self, labels, values, num_vertices, masks, ids, adm, train=False):
        file_to_write = self.train_results_file if train else self.valid_results_file
        with open(file_to_write, 'w') as out_file:
            self.print_all_results_as_graph(
                all_labels=labels, all_computed_values=values,
                all_num_vertices=num_vertices, all_masks=masks,
                all_ids=ids, all_adms=adm, out_file=out_file)

    def save_progress(self, model_path: str, train_step: int, valid_step: int) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "params": self.params,
            "weights": weights_to_save,
            "train_step": train_step,
            "valid_step": valid_step,
        }

        with open(model_path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                           tf.compat.v1.local_variables_initializer())

        self.sess.run(init_op)

    def restore_progress(self, model_path: str) -> (int, int):
        print("Restoring weights from file %s." % model_path)
        with open(model_path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            if self.args.get('--test_with_train') and par == 'valid_file':
                continue
            # Fine to have different task_ids:
            if par not in ['task_ids', 'num_epochs']:
                assert par_value == data_to_load['params'][par]


        variables_to_initialize = []
        with tf.compat.v1.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.compat.v1.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)

        return data_to_load['train_step'], data_to_load['valid_step']
