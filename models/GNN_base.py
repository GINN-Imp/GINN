#!/usr/bin/env/python

from typing import Tuple, List, Any, Sequence

import tensorflow as tf
import time
import os
import json
import numpy as np
import pickle
import random

from utils import MLP, ThreadedIterator, SMALL_NUMBER
import utils


class BaseModel(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 3000,
            'patience': 25,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            # updated according to the different representation.
            'hidden_size': 100,
            'num_timesteps': 4,
            'use_graph': True,
            # n% of the largest graph
            'on_large_data': 100,
            # n% of test result
            'splitTest': 0,

            'tie_fwd_bkwd': True,
            'task_ids': [0],

            'random_seed': 0,

            'train_file': 'molecules_train.json',
            'valid_file': 'molecules_valid.json',
            'test_file': '',
            'filterLabel': 0,
            'outputCSVPrefix': "",
            'move_to_test': ''
        }

    def moveToTest(self, train_data, test_data, projName):
        if projName == "":
            return
        utils.moveToTest(train_data, test_data, projName)
        print("after moving, #Train:#Test is %d:%d."%(len(train_data), len(test_data)))


    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)

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
        self.ops = {}

        if self.params["outputCSVPrefix"] == "":
            self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        else:
            self.log_file = os.path.join(log_dir, "%s_log.json" % self.params["outputCSVPrefix"])

        #with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            #json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])


        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.train_data = self.load_data(params['train_file'], is_training_data=True)
        self.valid_data = self.load_data(params['valid_file'], is_training_data=False)
        self.moveToTest(self.train_data, self.valid_data, params["move_to_test"])
        #utils.analyzeGraphInfo(self.train_data + self.valid_data)
        #exit()


        if params['test_file'] != "":
            self.test_data = self.load_data(params['test_file'], is_training_data=False)
        else:
            self.test_data = None

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.make_model()
            self.make_train_step()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        paths = full_path.split(";")
        data = []
        for full_path in paths:
            if not os.path.exists(full_path):
                continue
            with open(full_path, 'r') as f:
                tmp = json.load(f)
            data += tmp

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data:
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        return self.process_raw_graphs(data, is_training_data)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def addIntervalModel(self):
        return self.ops['final_node_representations']

    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['initial_node_representation'])

        self.ops['losses'] = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                self.computeLoss(task_id, internal_id)
        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])

    def calgraphstate(self):
        node_representations = self.ops['final_node_representations']
        node_features = self.placeholders['initial_node_representation']
        graphsize = self.placeholders['numOfNodesInGraph']
        graph_mask = tf.expand_dims(tf.sequence_mask(graphsize, dtype=tf.float32), -1)
        gate_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                name="node_gate_layer")
        output_layer = tf.keras.layers.Dense(node_representations.shape[-1],
                name="node_output_layer")
        # calculate weighted, node-level outputs
        node_all_repr = tf.concat([node_features, node_representations], axis=-1)
        graph_state = gate_layer(node_all_repr) * output_layer(node_representations)
        return graph_state

    def computeLoss(self, task_id, internal_id):
        self.computeLoss1(task_id, internal_id)

    def computePred(self, task_id, internal_id):
        with tf.variable_scope("regression_gate"):
            weights = {
                    'out': tf.Variable(tf.random_normal([self.params['hidden_size'], 2]))
                    }
            biases = {
                    'out': tf.Variable(tf.random_normal([2]))
                    }
        pred = tf.matmul(self.ops['final_node_representations'], weights['out']) + biases['out']
        # Sum up all nodes per-graph
        pred = tf.unsorted_segment_sum(data=pred,
                segment_ids=self.placeholders['graph_nodes_list'],
                num_segments=self.placeholders['num_graphs'])
        return pred
    def computeLoss1(self, task_id, internal_id):
        pred = self.computePred(task_id, internal_id)
        y = tf.to_int64(self.placeholders['target_values'][internal_id,:])
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y)
        pred = tf.argmax(pred,1)
        correct_pred = tf.equal(pred, y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.ops['accuracy_task%i' % task_id] = accuracy
        self.ops['losses'].append(cost)
        self.ops['acc_info'] = [pred, y, self.placeholders['numOfNodesInGraph']]

    def computeLoss2(self, task_id, internal_id):
        with tf.variable_scope("regression_gate"):
            self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
                    self.placeholders['out_layer_dropout_keep_prob'])
        with tf.variable_scope("regression"):
            self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
                    self.placeholders['out_layer_dropout_keep_prob'])
        computed_values = self.gated_regression(self.ops['final_node_representations'],
                self.weights['regression_gate_task%i' % task_id],
                self.weights['regression_transform_task%i' % task_id])
        diff = computed_values - self.placeholders['target_values'][internal_id,:]
        task_target_mask = self.placeholders['target_mask'][internal_id,:]
        task_target_num = tf.reduce_sum(task_target_mask) + SMALL_NUMBER
        diff = diff * task_target_mask  # Mask out unused values
        self.ops['accuracy_task%i' % task_id] = tf.reduce_sum(tf.abs(diff)) / task_target_num
        task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num
        # Normalise loss to account for fewer task-specific examples in batch:
        task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
        self.ops['losses'].append(task_loss)

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, epoch_name: str, data, is_training: bool):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])

        loss = 0
        accInfo = None
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops["acc_info"]]
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracies, acc_info) = (result[0], result[1], result[2])
            #print(result[2])
            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)
            if not is_training:
                accInfo = self.processAccInfo(accInfo, acc_info, num_graphs)

            print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
                                                                               step,
                                                                               num_graphs,
                                                                               loss / processed_graphs),
                  end='\r')

        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, accuracies, error_ratios, instance_per_sec, accInfo

    def processAccInfo(self, pre, new, num_graphs):
        if pre == None:
            pre = new
            return pre
        for i in range(len(pre)):
            pre[i] = np.concatenate((pre[i], new[i]), axis=-1)
        return pre

    def train(self):
        log_to_save = []
        best_accInfo = None
        total_time_start = time.time()
        total_training_time = 0
        total_validating_time = 0
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                _, valid_accs, _, _, _ = self.run_epoch("Resumed (validation)", self.valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (float("+inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                #print("== Epoch %i" % epoch)
                last_training_time = time.time()
                train_loss, train_accs, train_errs, train_speed, accInfo = self.run_epoch("epoch %i (training)" % epoch,
                        self.train_data, True)

                total_training_time += time.time() - last_training_time

                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                #print("\r\x1b[K Train: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (train_loss, accs_str, errs_str, train_speed))

                last_validating_time = time.time()
                valid_loss, valid_accs, valid_errs, valid_speed, accInfo = self.run_epoch("epoch %i (validation)" % epoch,
                        self.valid_data, False)

                total_validating_time += time.time() - last_validating_time

                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print("\r\x1b[K Valid: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (valid_loss, accs_str, errs_str, valid_speed))

                epoch_time = time.time() - total_time_start

                val_acc = np.sum(valid_accs)  # type: float
                # change val_acc < best_val_acc since change of accuracy.
                val_acc = 1-val_acc
                if val_acc < best_val_acc:
                    #self.save_model(self.best_model_file)
                    #print("  (Best epoch so far, cum. val. acc decreased to %.5f from %.5f. Saving to '%s')" % (val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                    best_accInfo = accInfo
                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params['patience'])
                    print("  (Best epoch %d, cum. val. acc is  %.5f)" % (best_val_acc_epoch, 1-best_val_acc))
                    if self.test_data:
                        self.performTest()
                    else:
                        self.parseAccInfo(best_accInfo)
                    break

                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': (0, 0, 0, 0),
                    'valid_results': utils.computeTop1F1(accInfo, self.params['filterLabel'])
                }

                log_to_save.append(log_entry)
                #with open(self.log_file, 'w') as f:
                    #json.dump(log_to_save, f, indent=4)
        print(" avg training time is %.1f s, avg pred time is %.1f s" % (total_training_time/epoch, total_validating_time/epoch))


    def performTest(self):
        if not self.test_data:
            return
        test_loss, test_accs, _, test_speed, accInfo = self.run_epoch("epoch (testing)" , self.test_data, False)

        # change val_acc < best_val_acc since change of accuracy.
        print("Found testing file, use the testing results for SE info. Testing accuracy is %.5f." % (test_accs))
        self.parseAccInfo(accInfo)


    def parseAccInfo(self, best_accInfo):
        TP, TN, FP, FN = utils.returnSEMet((best_accInfo[0],best_accInfo[1]))
        utils.computeF1(TP, TN, FP, FN)
        utils.returnSEMetByPortion(best_accInfo[0], best_accInfo[1], best_accInfo[2], self.params["splitTest"])

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
                         "params": self.params,
                         "weights": weights_to_save
                       }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different task_ids:
            if par not in ['task_ids', 'num_epochs']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)
