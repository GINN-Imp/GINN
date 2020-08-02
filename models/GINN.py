#!/usr/bin/env/python
"""
Usage:
    GINN.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
"""
from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict, namedtuple
import numpy as np
import tensorflow as tf
import sys, traceback, os, json
import pdb
import itertools

from GNN_base import BaseModel
from utils import MLP, glorot_init, SMALL_NUMBER
import utils

from tensor2tensor.models import transformer
from tensor2tensor.data_generators import problem_hparams



GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells',])


class IntervalGGNNModel(BaseModel):
    def __init__(self, args):
        self.transformer = None
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            #'batch_size': 100000,
            'batch_size': 200,
            'batch_graph_size': 0,
            'maxTargetLength': -1,
            'nodesThreshold': 1000,
            'rep_mode': 0,
            'use_attention': 0,
            'toLine': False,
            #'train_file': 'intervals-jfreechart-1.0.19.json',
            'train_file': 'intervals.json',
            'valid_file': 'intervals.json',
            'use_edge_bias': False,
            'use_propagation_attention': False,
            'use_edge_msg_avg_aggregation': True,
            'residual_connections': {  # For layer i, specify list of layers whose output is added as an input
                                     "2": [0],
                                     "4": [0, 2]
                                    },

            'layer_timesteps': [2, 2, 1, 2, 1],  # number of layers & propagation steps per layer

            'graph_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell, or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
            'iterSteps': 3,
            'edge_weight_dropout_keep_prob': .8
        })
        return params

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
        # this is the width of feature encoding.
        isDifferent = False
        isProcessed = False
        if self.params.get("maxWidth") == None:
            self.params["maxWidth"] = 0
        else:
            isProcessed = True


        tmp = []
        # this is used to drop complex CFGs.
        nodesThreshold = self.params["nodesThreshold"]
        if self.params["on_large_data"] != 100:
            nodesThreshold = 100000
        numOfNodesDropped = 0
        self.ops['maxTargetLen'] = 0
        for intervals in data:
            for key in intervals:
                if not key.isdigit():
                    # possible cases:
                    # targets, insideinterval, numOfNode, graph, bugPos
                    if key == "graph":
                        num_fwd_edge_types = max(num_fwd_edge_types,
                            max([e[1] for e in intervals[key]]))
                    continue

                g = intervals[key]
                if g.get("node_features") != None:
                    for r in g["node_features"]:
                        newR = 0
                        for index in range(len(r)):
                            newR += r[index]
                        if self.params["maxWidth"] < newR:
                            if isProcessed:
                                isDifferent = True
                            self.params["maxWidth"] = newR
                            tmp = r
                self.max_num_vertices = max(self.max_num_vertices,
                        max([v for e in g['graph'] for v in [e[0], e[2]]]))
                num_fwd_edge_types = max(num_fwd_edge_types,
                        max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size, len(data[0]["0"]["node_features"][0]))
        self.number_of_tokens = len(data[0]["0"]["node_features"][0])

        # begin process_raw_graphs
        processed_graphs = []
        totalNumOfIntervals = 0
        totalNumOfWholeGraphs = 0
        totalNumOfNodes = 0
        for intervals in data:
            intervalGraph = []
            for key in intervals:
                if key == "targets":
                    intervalTarget = intervals[key]
                    continue
                elif key == "insideinterval":
                    continue
                elif key == "numOfNode":
                    numOfNode = intervals[key]
                    continue
                elif key == "graph":
                    (intervalAdjLists, intervalNIEPT) = self.__graph_to_adjacency_lists(intervals[key])
                    continue
                elif key == "bugPos":
                    intervalBugPos = intervals[key]
                    continue
                elif key == "fileHash":
                    intervalFileHash = intervals[key]
                    continue
                elif key == "projName":
                    intervalProjName = intervals[key]
                    continue
                elif key == "funName":
                    intervalFunName = intervals[key]
                    continue
                d = intervals[key]
                (adjacency_lists, num_incoming_edge_per_type) = self.__graph_to_adjacency_lists(d['graph'])
                #convertedRep, sl = self.convertRepresentation(d["node_features"])
                #for ind, n in enumerate(d["numOfFeatures"]):
                    #assert n == sl[ind],"Inconsistent features."
                totalNumOfNodes += len(d["node_features"])

                assert len(d["node_features"]) != 0

                intervalGraph.append({"adjacency_lists": adjacency_lists,
                    "num_incoming_edge_per_type": num_incoming_edge_per_type,
                    "insideinterval": 1,
                    "graphIndex": int(key),
                    "init": d["node_features"],
                    "nodeMask": d["node_mask"],
                    "convRep": self.convertListToNumpy(d["convRep"]),
                    "bugPos": d["bugPos"],
                    "sl":d["numOfFeatures"]})

            intervalGraph = sorted(intervalGraph, key=lambda k: k['graphIndex'])


            if numOfNode > nodesThreshold:
                # drop complex nodes
                numOfNodesDropped += 1
                continue

            labels = [intervalTarget[task_id] for task_id in self.params['task_ids']]
            if len(labels) > 0 and len(labels[0]) > self.ops["maxTargetLen"]:
                self.ops["maxTargetLen"] = len(labels[0])
            if numOfNode == 0:
                continue
            intervalGraph.append({"adjacency_lists": intervalAdjLists,
                "num_incoming_edge_per_type": intervalNIEPT,
                "bugPos": intervalBugPos,
                "insideinterval": 0,
                "graphIndex": -1,
                "numOfNode": numOfNode,
                "fileHash": str(intervalFileHash[0])+"-"+intervalFunName,
                "projName": intervalProjName,
                "labels": labels})
            processed_graphs.append(intervalGraph)
            totalNumOfIntervals += numOfNode
            totalNumOfWholeGraphs += 1

        print("total graph: %d and average intervals per graph: %f, aveNodes: %f, maxWidth is: %d, droppedNodes: %d"
                %( totalNumOfWholeGraphs, float(totalNumOfIntervals)/totalNumOfWholeGraphs, float(totalNumOfNodes)/totalNumOfWholeGraphs, self.params["maxWidth"], numOfNodesDropped))


        if is_training_data:
            np.random.shuffle(processed_graphs)
            for task_id in self.params['task_ids']:
                task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                if task_sample_ratio is not None:
                    ex_to_sample = int(len(processed_graphs) * task_sample_ratio)
                    for ex_id in range(ex_to_sample, len(processed_graphs)):
                        processed_graphs[ex_id]['labels'][task_id] = None

        if isDifferent:
            for d in self.train_data:
                for interval in d:
                    if interval.get("convRep") != None:
                        interval["convRep"] = self.convertListToNumpy(interval["convRep"])


        #TODO: added to eval performance on a specific percent of data
        if is_training_data:
            utils.filterGraphByPerc(processed_graphs, self.params['on_large_data'])
            print("after filtering, remain %d graphs."%(len(processed_graphs)))
        return processed_graphs

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, h_dim],
                                                                          name='node_features')
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.num_edge_types)]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.num_edge_types],
                                                                          name='num_incoming_edges_per_type')
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='graph_nodes_list')
        self.placeholders['numOfNodesInGraph'] = tf.placeholder(tf.int32, [None], name='numOfNodesInGraph')
        self.placeholders['numOfNodesInGraphsPad'] = tf.placeholder(tf.float32, [None], name='numOfNodesInGraphsPad')
        self.placeholders['attentionWeight'] = tf.placeholder(tf.float32, [None, h_dim],
                name='attentionWeight')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')

        self.placeholders['batch_SL'] = tf.placeholder(tf.int32, [None], name='batch_SL')
        self.placeholders['targetLength'] = tf.placeholder(tf.int32, [None], name='targetLength')
        self.placeholders['interIntervalNodeLabels'] = tf.placeholder(tf.int32, [None], name='interIntervalNodeLabels')
        self.placeholders['intraIntervalNodeLabels'] = tf.placeholder(tf.int32, [None], name='intraIntervalNodeLabels')
        self.placeholders['interIntervalNodeLabelsIndex'] = tf.placeholder(tf.int32, [None], name='interIntervalNodeLabelsIndex')
        self.placeholders['intraIntervalNodeLabelsIndex'] = tf.placeholder(tf.int32, [None], name='intraIntervalNodeLabelsIndex')
        self.placeholders['converted_node_representation'] = tf.placeholder(tf.int32,
                [None, self.params["maxWidth"]],
                name='convertedNodeRep')

        self.placeholders['numIntervalGraphs'] = tf.placeholder(tf.int32, [],
                name='numIntervalGraphs')
        self.placeholders['numIntraGraphs'] = tf.placeholder(tf.int32, [],
                name='numIntraGraphs')
        self.placeholders['numOfValidNodesInGraph'] = tf.placeholder(tf.int32, [None],
                name='numOfValidNodesInGraph')
        self.placeholders['fileHash'] = tf.placeholder(tf.string, [None],
                name='fileHash')
        self.placeholders['nodeMask'] = tf.placeholder(tf.bool, [None],
                name='nodeMask')
        self.placeholders['numOfNodeInSubgraph'] = tf.placeholder(tf.float32, [None, h_dim], name='numOfNodeInSubgraph')
        self.placeholders['intervalGraphNodesList'] = tf.placeholder(tf.int32, [None], name='intervalGraphNodesList')
        self.placeholders['interLabelMasks'] = tf.placeholder(tf.bool, [None], name='interLabelMasks')
        self.placeholders['graphLabelMasks'] = tf.placeholder(tf.bool, [None], name='graphLabelMasks')
        self.placeholders['intraLabelMasks'] = tf.placeholder(tf.bool, [None], name='intraLabelMasks')
        self.placeholders['intraValidLabelMasks'] = tf.placeholder(tf.bool, [None], name='intraValidLabelMasks')
        self.placeholders['intervalNumIncomingEdgesPerType'] =tf.placeholder(tf.float32,
                [None, self.num_edge_types], name='intervalNumIncomingEdgesPerType')
        self.placeholders['intervalAdjLists'] = [tf.placeholder(tf.int32, [None, 2], name='intervalAdj_e%s' % e)
                for e in range(self.num_edge_types)]
        self.placeholders['interLabelIndex'] = tf.placeholder(tf.int32,
                [None, None],
                name='interLabelIndex')
        self.placeholders['intraLabelIndex'] = tf.placeholder(tf.int32,
                [None, None],
                name='intraLabelIndex')



        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units:
        self.weights = {}  # Used by super-class to place generic things
        self.gnn_weights = GGNNWeights([], [], [], [])
        for layer_idx in range(len(self.params['layer_timesteps'])):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                edge_weights = tf.Variable(glorot_init([self.num_edge_types * h_dim, h_dim]),
                                           name='gnn_edge_weights_%i' % layer_idx)
                edge_weights = tf.reshape(edge_weights, [self.num_edge_types, h_dim, h_dim])
                edge_weights = tf.nn.dropout(edge_weights, keep_prob=self.placeholders['edge_weight_dropout_keep_prob'])
                self.gnn_weights.edge_weights.append(edge_weights)

                if self.params['use_propagation_attention']:
                    self.gnn_weights.edge_type_attention_weights.append(tf.Variable(np.ones([self.num_edge_types], dtype=np.float32),
                                                                                    name='edge_type_attention_weights_%i' % layer_idx))

                if self.params['use_edge_bias']:
                    self.gnn_weights.edge_biases.append(tf.Variable(np.zeros([self.num_edge_types, h_dim], dtype=np.float32),
                                                                    name='gnn_edge_biases_%i' % layer_idx))

                cell_type = self.params['graph_rnn_cell'].lower()
                if cell_type == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
                elif cell_type == 'cudnncompatiblegrucell':
                    assert(activation_name == 'tanh')
                    import tensorflow.contrib.cudnn_rnn as cudnn_rnn
                    cell = cudnn_rnn.CudnnCompatibleGRUCell(h_dim)
                elif cell_type == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
                else:
                    raise Exception("Unknown RNN cell type '%s'." % cell_type)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     state_keep_prob=self.placeholders['graph_state_keep_prob'])
                self.gnn_weights.rnn_cells.append(cell)


    def message_computing(self, node_states, message_targets, num_nodes, layer_idx, layer_residual_states, adjLists, numIncomingEdgesPerType):
        messages = []
        message_source_states = []
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjLists):
            edge_sources = adjacency_list_for_edge_type[:, 0]
            edge_source_states = tf.nn.embedding_lookup(params=node_states,
                    ids=edge_sources)  # Shape [E, D]
            all_messages_for_edge_type = tf.matmul(edge_source_states,
                    self.gnn_weights.edge_weights[layer_idx][edge_type_idx])  # Shape [E, D]
            messages.append(all_messages_for_edge_type)
            message_source_states.append(edge_source_states)
        messages = tf.concat(messages, axis=0)
        if self.params['use_propagation_attention']:
            message_source_states = tf.concat(message_source_states, axis=0)  # Shape [M, D]
            message_target_states = tf.nn.embedding_lookup(params=node_states,
                    ids=message_targets)  # Shape [M, D]
            message_attention_scores = tf.einsum('mi,mi->m', message_source_states, message_target_states)  # Shape [M]
            message_attention_scores = message_attention_scores * message_edge_type_factors
            # The following is softmax-ing over the incoming messages per node.
            # As the number of incoming varies, we can't just use tf.softmax. Reimplement with logsumexp trick:
            # Step (1): Obtain shift constant as max of messages going into a node
            message_attention_score_max_per_target = tf.unsorted_segment_max(data=message_attention_scores,
                    segment_ids=message_targets,
                    num_segments=num_nodes)  # Shape [V]
            # Step (2): Distribute max out to the corresponding messages again, and shift scores:
            message_attention_score_max_per_message = tf.gather(params=message_attention_score_max_per_target,
                    indices=message_targets)  # Shape [M]
            message_attention_scores -= message_attention_score_max_per_message
            # Step (3): Exp, sum up per target, compute exp(score) / exp(sum) as attention prob:
            message_attention_scores_exped = tf.exp(message_attention_scores)  # Shape [M]
            message_attention_score_sum_per_target = tf.unsorted_segment_sum(data=message_attention_scores_exped,
                    segment_ids=message_targets,
                    num_segments=num_nodes) # Shape [V]
            message_attention_normalisation_sum_per_message = tf.gather(
                    params=message_attention_score_sum_per_target,
                    indices=message_targets) # Shape [M]
            message_attention = message_attention_scores_exped / (message_attention_normalisation_sum_per_message
                    + SMALL_NUMBER)  # Shape [M]
            # Step (4): Weigh messages using the attention prob:
            messages = messages * tf.expand_dims(message_attention, -1)
        incoming_messages = tf.unsorted_segment_sum(data=messages,
                segment_ids=message_targets,
                num_segments=num_nodes)

        if self.params['use_edge_bias']:
            incoming_messages += tf.matmul(numIncomingEdgesPerType,
                    self.gnn_weights.edge_biases[layer_idx])  # Shape [V, D]

        if self.params['use_edge_msg_avg_aggregation']:
            num_incoming_edges = tf.reduce_sum(numIncomingEdgesPerType,
                    keep_dims=True, axis=-1)  # Shape [V, 1]
            incoming_messages /= num_incoming_edges + SMALL_NUMBER
        incoming_information = tf.concat(layer_residual_states + [incoming_messages],
                axis=-1)  # Shape [V, D*(1 + num of residual connections)]

        # pass updated vertex features into RNN cell
        return self.gnn_weights.rnn_cells[layer_idx](incoming_information,
                node_states)[1]

    def _compute_final_node_representations_inside(self, nodeRep, adjLists, numIncomingEdgesPerType) -> tf.Tensor:
        node_states_per_layer = []  # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer.append(nodeRep)
        node_state_interval_per_layer = [] # for upper-graph
        num_nodes = tf.shape(nodeRep, out_type=tf.int32)[0]
        num_nodes_upper = self.placeholders['numIntraGraphs']

        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_types = []  # list of tensors of edge type of shape [E]
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjLists):
            edge_targets = adjacency_list_for_edge_type[:, 1]
            message_targets.append(edge_targets)
            message_edge_types.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
        message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

        message_targets_upper = []
        message_edge_types_upper = []
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['intervalAdjLists']):
            edge_targets = adjacency_list_for_edge_type[:, 1]
            message_targets_upper.append(edge_targets)
            message_edge_types_upper.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
        message_targets_upper = tf.concat(message_targets_upper, axis=0)
        message_edge_types_upper = tf.concat(message_edge_types_upper, axis=0)

        retRep = None
        for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                # Used shape abbreviations:
                #   V ~ number of nodes
                #   D ~ state dimension
                #   E ~ number of edges of current type
                #   M ~ number of messages (sum of all E)

                # Extract residual messages, if any:
                layer_residual_connections = self.params['residual_connections'].get(str(layer_idx))
                if layer_residual_connections is None:
                    layer_residual_states = []
                    layer_residual_states_upper = []
                else:
                    layer_residual_states = [node_states_per_layer[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]
                    layer_residual_states_upper = [node_state_interval_per_layer[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]

                # Record new states for this layer. Initialised to last state, but will be updated below:
                node_states_per_layer.append(node_states_per_layer[-1])
                for step in range(num_timesteps):
                    with tf.variable_scope('timestep_%i' % step):

                        node_states = self.message_computing(node_states_per_layer[-1],
                                message_targets, num_nodes, layer_idx,
                                layer_residual_states, adjLists,
                                self.placeholders['num_incoming_edges_per_type'])
                        #merge them into nodes for intervals.
                        node_states = tf.math.multiply(node_states, self.ops['attentionWeight'])

                        # for return states, should not merged by averaging
                        retRep = node_states

                        node_state_interval_per_layer.append(tf.math.unsorted_segment_sum(
                            node_states,
                            segment_ids=self.placeholders['graph_nodes_list'],
                            num_segments=self.placeholders['numIntraGraphs'])
                            ) # Shape [Ve, D]

                        #node_states_per_layer[-1] = node_states
                        #continue
                        node_states = self.message_computing(node_state_interval_per_layer[-1],
                                message_targets_upper, num_nodes_upper, layer_idx,
                                layer_residual_states_upper, self.placeholders['intervalAdjLists'],
                                self.placeholders['intervalNumIncomingEdgesPerType'])
                        #if (self.ops['initialRepresentation'] == None):
                            #self.ops['initialRepresentation'] = node_states
                        #node_states = tf.div(node_states, self.placeholders['numOfNodeInSubgraph'])
                        #node_states = tf.nn.embedding_lookup(node_states,
                        #        self.placeholders['graph_nodes_list']) # Shape [V, D]
                        #node_states_per_layer[-1] = node_states
                        node_states = tf.nn.embedding_lookup(node_states,
                                self.placeholders['graph_nodes_list']) # Shape [V, D]
                        node_states = tf.math.multiply(node_states, self.ops['attentionWeight'])
                        numOfNodesInGraphsPad = self.placeholders['numOfNodesInGraphsPad']
                        numOfNodesInGraphsPad = tf.expand_dims(numOfNodesInGraphsPad, -1)
                        node_states = tf.math.multiply(node_states, numOfNodesInGraphsPad)
                        node_states_per_layer[-1] = node_states

        return retRep
        #return node_states_per_layer[-1]

    def computePred(self, task_id, internal_id):
        with tf.variable_scope("regression_gate"):
            self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 2,
                    [], self.placeholders['out_layer_dropout_keep_prob'])
        with tf.variable_scope("regression"):
            self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 2,
                    [], self.placeholders['out_layer_dropout_keep_prob'])

        pred = self.gated_regression(self.ops['final_node_representations'],
                self.weights['regression_gate_task%i' % task_id],
                self.weights['regression_transform_task%i' % task_id])

        return pred

    def calAttention(self):
        finalGraphRep = self.ops["final_node_representations"]
        finalGraphRep = tf.unsorted_segment_sum(data=finalGraphRep,
                segment_ids=self.ops['nodeListsForGated'],
                num_segments=self.ops['numGraphForGated'])  # [g x hidden_size]
        if self.ops.get('nodeListsForGated2') != None:
            finalGraphRep = tf.unsorted_segment_sum(data=finalGraphRep,
                    segment_ids=self.ops['nodeListsForGated2'],
                    num_segments=self.ops['numGraphForGated2'])  # [g x hidden_size]
        finalGraphRep = tf.nn.embedding_lookup(finalGraphRep, self.placeholders['intervalGraphNodesList'])
        finalGraphRep = tf.nn.embedding_lookup(finalGraphRep, self.placeholders['graph_nodes_list'])

        x = self.placeholders['converted_node_representation']
        embeddings = tf.get_variable('embedding_matrix', [self.number_of_tokens, self.params["hidden_size"]])
        x = tf.nn.embedding_lookup(embeddings, x) # batch_size*maxWidth*hidden_size
        # Get the number of rows in the fed value at run-time.
        ph_num_rows = tf.shape(x)[1]
        finalGraphRep = tf.expand_dims(finalGraphRep, axis=1) # Add dimension
        finalGraphRep = tf.tile(finalGraphRep, multiples=tf.stack([1,ph_num_rows,1])) # Duplicate in this dimension
        con_x = tf.concat([x,finalGraphRep], axis=-1) # Concatenate on innermost dimension


        with tf.variable_scope("feedforward"):
            weights = {
                    'out': tf.Variable(tf.random_normal([2*self.params['hidden_size'], 1]))
                    }
            biases = {
                    'out': tf.Variable(tf.random_normal([1]))
                    }

        C = tf.tensordot(con_x, weights['out'], axes=1) + biases['out'] # batch_size*maxWidth*1
        C = tf.squeeze(C, axis=2)
        #TODO: why axis=1 is better than axis=0 in simple-conf??
        C = tf.nn.softmax(C, axis=0)
        # This is the overall attention socre
        self.ops['attentionScore'] = C
        return x

    def computeAttentionWeight(self):
        partition = tf.math.unsorted_segment_sum(
                self.ops['attentionScore'],
                segment_ids=self.placeholders['graph_nodes_list'],
                num_segments=self.placeholders['numIntraGraphs'])
        # used to cal attention in each interval.
        # This is the attention socre in each interval compare to attentionScore
        partition = tf.nn.embedding_lookup(partition,
                self.placeholders['intervalGraphNodesList'])
        self.ops['attentionWeight'] = tf.div(self.ops['attentionScore'], partition)

    def computeLoss1(self, task_id, internal_id):
        if self.params["use_attention"]:
            self.calAttention()
            self.computeAttentionWeight()
        pred = self.computePred(task_id, internal_id)
        y = tf.to_int64(self.placeholders['target_values'][internal_id,:])

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y)
        pred = tf.argmax(pred,1)
        correct_pred = tf.equal(pred, y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #self.ops['pred'] = tf.argmax(pred,1)
        self.ops['accuracy_task%i' % task_id] = accuracy
        self.ops['losses'].append(cost)
        self.ops['acc_info'] = [pred, y, self.placeholders['numOfNodesInGraph']]

    def computeLoss2(self, task_id, internal_id):
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
                segment_ids=self.ops['nodeListsForGated'],
                num_segments=self.ops['numGraphForGated'])  # [g x 2]
        if self.ops.get('nodeListsForGated2') != None:
            pred = tf.unsorted_segment_sum(data=pred,
                    segment_ids=self.ops['nodeListsForGated2'],
                    num_segments=self.ops['numGraphForGated2'])  # [g x 2]
        y = tf.to_int64(self.placeholders['target_values'][internal_id,:])
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y)
        pred = tf.argmax(pred,1)
        correct_pred = tf.equal(pred, y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #self.ops['pred'] = tf.argmax(pred,1)
        self.ops['accuracy_task%i' % task_id] = accuracy
        self.ops['losses'].append(cost)
        self.ops['acc_info'] = [pred, y, self.placeholders['numOfNodesInGraph']]

    def computeLoss(self, task_id, internal_id):
        self.computeLoss1(task_id, internal_id)
        return
        #self.computeLoss2(task_id, internal_id)
        #super().computeLoss(task_id, internal_id)
        #return

    def compute_final_node_representations_inside(self):
        #self.ops['nodeListsForGated'] = self.placeholders['graph_nodes_list']
        #self.ops['numGraphForGated'] = self.placeholders['numIntraGraphs']
        # the output is still based on the upper graph.
        self.ops['nodeListsForGated'] = self.placeholders['graph_nodes_list']
        self.ops['numGraphForGated'] = self.placeholders['numIntraGraphs']
        self.ops['nodeListsForGated2'] = self.placeholders['intervalGraphNodesList']
        self.ops['numGraphForGated2'] = self.placeholders['numIntervalGraphs']
        # graph_nodes_list: [0,0,1,1,1,2,2,2,2,2], nodes with same labels are in the same interval.
        # attentionWeight: [0.5, 0.5, 0.333,0.333,0.333,0.2,0.2,0.2,0.2,0.2]
        self.ops['attentionWeight'] = self.placeholders['attentionWeight']
        #self.ops['attentionWeight'] = tf.expand_dims(self.ops['attentionWeight'], 1)

        # TODO: the initialRepresentation should be initial_node_representation, however, it can't due to
        # the label of graphs.
        self.ops['initialRepresentation'] = self.placeholders['initial_node_representation']
        finalNodeRepresentation = self._compute_final_node_representations_inside(
                self.placeholders['initial_node_representation'],
                self.placeholders['adjacency_lists'],
                self.placeholders['num_incoming_edges_per_type'],) # Shape [V, D]
        return finalNodeRepresentation

    def convertInitialNodeRep(self):
        x = self.placeholders['converted_node_representation']
        #TODO: not sure about the embedding size
        embeddings = tf.get_variable('embedding_matrix', [self.number_of_tokens, self.params["hidden_size"]])
        x = tf.nn.embedding_lookup(embeddings, x)
        #TODO: not sure about the hidden_size
        trace_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        sl = self.placeholders['batch_SL']
        #rnn_state = trace_encoder_cell.zero_state(self.params['batch_size'], tf.float32)
        #dyoutputs, dyfinalstate = tf.nn.dynamic_rnn(trace_encoder_cell, x, initial_state=rnn_state, sequence_length=sl)
        dyoutputs, dyfinalstate = tf.nn.dynamic_rnn(trace_encoder_cell, x, sequence_length=sl, dtype=tf.float32)
        self.placeholders['initial_node_representation'] = dyfinalstate[1]


    def getModel(self, data):
        if self.transformer == None:
            hparams = transformer.transformer_small()
            #hparams = transformer.transformer_base()
            hparams.batch_size = self.params["batch_size"]
            hparams.hidden_size = self.params["hidden_size"]
            p_hparams = problem_hparams.test_problem_hparams(self.number_of_tokens, 100, hparams)
            #TODO train?
            #self.transformer = transformer.Transformer(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
            self.transformer = transformer.TransformerEncoder(hparams, mode=tf.estimator.ModeKeys.TRAIN)
        return self.transformer

    def convertTransformers(self):
        x = self.placeholders['converted_node_representation']
        #TODO: not sure about the embedding size
        embeddings = tf.get_variable('embedding_matrix', [self.number_of_tokens, self.params["hidden_size"]])
        x = tf.nn.embedding_lookup(embeddings, x)

        x = tf.expand_dims(x, 2)
        model = self.getModel(x)
        features = {
                "inputs" : x,
                "targets": 1,
                "target_space_id": 0
                }

        #output, _ = model.encode(features["inputs"], 1, model._hparams)
        output = model(features)
        x = output[0]
        x = tf.nn.max_pool(x, [1,self.params["maxWidth"],1,1], [1,self.params["maxWidth"],1,1], padding='VALID')
        x = tf.squeeze(x, 1)
        x = tf.squeeze(x, 1)
        #paddings = tf.constant([[0, 0,], [0, 68]])
        #x = tf.pad(x, paddings)
        self.placeholders['initial_node_representation'] = x

    def TransformerNLSTM(self):
        x = self.placeholders['converted_node_representation']
        #TODO: not sure about the embedding size
        embeddings = tf.get_variable('embedding_matrix', [self.number_of_tokens, self.params["hidden_size"]])
        x = tf.nn.embedding_lookup(embeddings, x)

        x = tf.expand_dims(x, 2)
        model = self.getModel(x)
        features = {
                "inputs" : x,
                "targets": 1,
                "target_space_id": 0
                }

        output = model(features)
        x = output[0]
        x = tf.squeeze(x, 2)

        trace_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        #rnn_state = trace_encoder_cell.zero_state(self.params['batch_size'], tf.float32)
        #dyoutputs, dyfinalstate = tf.nn.dynamic_rnn(trace_encoder_cell, x, initial_state=rnn_state, sequence_length=sl)
        dyoutputs, dyfinalstate = tf.nn.dynamic_rnn(trace_encoder_cell, x, dtype=tf.float32)
        self.placeholders['initial_node_representation'] = dyfinalstate[1]

    def compute_final_node_representations(self):
        RepMode = self.params["rep_mode"]
        if RepMode == 1 or RepMode == 4:
            self.convertInitialNodeRep()
        elif RepMode == 2:
            self.convertTransformers()
        elif RepMode == 3:
            self.TransformerNLSTM()
        #return self.compute_final_node_representations_outside()
        return self.compute_final_node_representations_inside()

    def compute_final_node_representations_outside(self):
        finalNodeRepresentation = self.placeholders['initial_node_representation']
        self.ops['nodeListsForGated'] = self.placeholders['intervalGraphNodesList']
        self.ops['numGraphForGated'] = self.placeholders['numIntervalGraphs']
        count = 0
        while(True):
            self.ops['intervalNodeRepresentation'] = self._compute_final_node_representations(
                    finalNodeRepresentation,
                    self.placeholders['adjacency_lists'],
                    self.placeholders['num_incoming_edges_per_type']) # Shape [V, D]
            self.ops['intervalNodeRepresentation'] = tf.math.unsorted_segment_sum(
                    self.ops['intervalNodeRepresentation'],
                    segment_ids=self.placeholders['graph_nodes_list'],
                    num_segments=self.placeholders['numIntraGraphs']) # Shape [Ve, D]
            if self.ops.get('initialRepresentation') == None:
                self.ops['initialRepresentation'] = self.ops['intervalNodeRepresentation']
            finalNodeRepresentation = self._compute_final_node_representations(
                    self.ops['intervalNodeRepresentation'],
                    self.placeholders['intervalAdjLists'],
                    self.placeholders['intervalNumIncomingEdgesPerType']) # Shape [Ve, D]
            if count >= self.params['iterSteps']:
                return finalNodeRepresentation
            count += 1
            finalNodeRepresentation = tf.div(finalNodeRepresentation, self.placeholders['numOfNodeInSubgraph'])
            finalNodeRepresentation = tf.nn.embedding_lookup(finalNodeRepresentation,
                    self.placeholders['graph_nodes_list']) # Shape [V, D]

        return finalNodeRepresentation

    def _compute_final_node_representations(self, nodeRep, adjLists, numIncomingEdgesPerType) -> tf.Tensor:
        node_states_per_layer = []  # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer.append(nodeRep)
        num_nodes = tf.shape(nodeRep, out_type=tf.int32)[0]

        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_types = []  # list of tensors of edge type of shape [E]
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjLists):
            edge_targets = adjacency_list_for_edge_type[:, 1]
            message_targets.append(edge_targets)
            message_edge_types.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
        message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

        for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                # Used shape abbreviations:
                #   V ~ number of nodes
                #   D ~ state dimension
                #   E ~ number of edges of current type
                #   M ~ number of messages (sum of all E)

                # Extract residual messages, if any:
                layer_residual_connections = self.params['residual_connections'].get(str(layer_idx))
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    layer_residual_states = [node_states_per_layer[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]

                if self.params['use_propagation_attention']:
                    message_edge_type_factors = tf.nn.embedding_lookup(params=self.gnn_weights.edge_type_attention_weights[layer_idx],
                                                                       ids=message_edge_types)  # Shape [M]

                # Record new states for this layer. Initialised to last state, but will be updated below:
                node_states_per_layer.append(node_states_per_layer[-1])
                for step in range(num_timesteps):
                    with tf.variable_scope('timestep_%i' % step):
                        messages = []  # list of tensors of messages of shape [E, D]
                        message_source_states = []  # list of tensors of edge source states of shape [E, D]

                        # Collect incoming messages per edge type
                        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjLists):
                            edge_sources = adjacency_list_for_edge_type[:, 0]
                            edge_source_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                        ids=edge_sources)  # Shape [E, D]
                            all_messages_for_edge_type = tf.matmul(edge_source_states,
                                                                   self.gnn_weights.edge_weights[layer_idx][edge_type_idx])  # Shape [E, D]
                            messages.append(all_messages_for_edge_type)
                            message_source_states.append(edge_source_states)

                        messages = tf.concat(messages, axis=0)  # Shape [M, D]

                        if self.params['use_propagation_attention']:
                            message_source_states = tf.concat(message_source_states, axis=0)  # Shape [M, D]
                            message_target_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                           ids=message_targets)  # Shape [M, D]
                            message_attention_scores = tf.einsum('mi,mi->m', message_source_states, message_target_states)  # Shape [M]
                            message_attention_scores = message_attention_scores * message_edge_type_factors

                            # The following is softmax-ing over the incoming messages per node.
                            # As the number of incoming varies, we can't just use tf.softmax. Reimplement with logsumexp trick:
                            # Step (1): Obtain shift constant as max of messages going into a node
                            message_attention_score_max_per_target = tf.unsorted_segment_max(data=message_attention_scores,
                                                                                             segment_ids=message_targets,
                                                                                             num_segments=num_nodes)  # Shape [V]
                            # Step (2): Distribute max out to the corresponding messages again, and shift scores:
                            message_attention_score_max_per_message = tf.gather(params=message_attention_score_max_per_target,
                                                                                indices=message_targets)  # Shape [M]
                            message_attention_scores -= message_attention_score_max_per_message
                            # Step (3): Exp, sum up per target, compute exp(score) / exp(sum) as attention prob:
                            message_attention_scores_exped = tf.exp(message_attention_scores)  # Shape [M]
                            message_attention_score_sum_per_target = tf.unsorted_segment_sum(data=message_attention_scores_exped,
                                                                                             segment_ids=message_targets,
                                                                                             num_segments=num_nodes)  # Shape [V]
                            message_attention_normalisation_sum_per_message = tf.gather(params=message_attention_score_sum_per_target,
                                                                                        indices=message_targets)  # Shape [M]
                            message_attention = message_attention_scores_exped / (message_attention_normalisation_sum_per_message + SMALL_NUMBER)  # Shape [M]
                            # Step (4): Weigh messages using the attention prob:
                            messages = messages * tf.expand_dims(message_attention, -1)

                        incoming_messages = tf.unsorted_segment_sum(data=messages,
                                                                    segment_ids=message_targets,
                                                                    num_segments=num_nodes)  # Shape [V, D]

                        if self.params['use_edge_bias']:
                            incoming_messages += tf.matmul(numIncomingEdgesPerType,
                                                           self.gnn_weights.edge_biases[layer_idx])  # Shape [V, D]

                        if self.params['use_edge_msg_avg_aggregation']:
                            num_incoming_edges = tf.reduce_sum(numIncomingEdgesPerType,
                                                               keep_dims=True, axis=-1)  # Shape [V, 1]
                            incoming_messages /= num_incoming_edges + SMALL_NUMBER

                        incoming_information = tf.concat(layer_residual_states + [incoming_messages],
                                                         axis=-1)  # Shape [V, D*(1 + num of residual connections)]

                        # pass updated vertex features into RNN cell
                        node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_idx](incoming_information,
                                                                                          node_states_per_layer[-1])[1]  # Shape [V, D]
        return node_states_per_layer[-1]

    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [v x 1]

        gated_outputs = tf.unsorted_segment_sum(data=gated_outputs,
                segment_ids=self.ops['nodeListsForGated'],
                num_segments=self.ops['numGraphForGated'])  # [g x 2]
        if self.ops.get('nodeListsForGated2') != None:
            gated_outputs = tf.unsorted_segment_sum(data=gated_outputs,
                    segment_ids=self.ops['nodeListsForGated2'],
                    num_segments=self.ops['numGraphForGated2'])  # [g x 2]

        return gated_outputs  # [g]

    def __graph_to_adjacency_lists(self, graph) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
        adj_lists = defaultdict(list)
        num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
        for src, e, dest in graph:
            fwd_edge_type = e - 1  # Make edges start from 0
            adj_lists[fwd_edge_type].append((src, dest))
            num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1
            if self.params['tie_fwd_bkwd']:
                adj_lists[fwd_edge_type].append((dest, src))
                num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params['tie_fwd_bkwd']):
            for (edge_type, edges) in adj_lists.items():
                bwd_edge_type = self.num_edge_types + edge_type
                final_adj_lists[bwd_edge_type] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)
                for (x, y) in edges:
                    num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

        return final_adj_lists, num_incoming_edges_dicts_per_type

    def convertListToNumpy(self, rep):
        limitmaxWidth = True
        if limitmaxWidth:
            self.params["maxWidth"] = 100
        maxLength = self.params["maxWidth"]
        for index in range(len(rep)):
            #assert len(rep[index]) <= maxLength, "ERROR in length."
            if limitmaxWidth:
                if len(rep[index]) > maxLength:
                    rep[index] = rep[index][:maxLength]
            if len(rep[index]) < maxLength:
                rep[index].extend([0]*(maxLength-len(rep[index])))
        return rep

    def isBuggyMethod(self, g):
        for interval in g:
            if interval.get("insideinterval") == 1:
                if 1 in interval["bugPos"]:
                    return True
            else:
                if interval["labels"][0] == 0.0:
                    return False
        return False

    def convertRepresentation(self, rep):
        #[[1,1,2,1], [0, 0, 1, 1]] => [[0,1,2,2,3], [2, 3]], [5, 2]
        newRep = []
        sl = []
        maxLength = self.params["maxWidth"]
        tmp = None
        for r in rep:
            newR = []
            for index in range(len(r)):
                newR.extend([index]*r[index])
            sl.append(len(newR))
            if len(newR) < maxLength:
                newR.extend([0]*(maxLength-len(newR)))
            newRep.append(np.array(newR))
        #newRep=np.array(list(itertools.zip_longest(*newRep, fillvalue=0))).T
        return newRep, sl

    def batchCond(self, intervalGraphIndex, data, num_graphs, node_offset, numOfGraphs):
        #num_graphs includes intervals and numOfGraphs doesn't
        if intervalGraphIndex < len(data) and (num_graphs == 0 or ((node_offset + len(data[intervalGraphIndex][0]['init']) * len(data[intervalGraphIndex])) < self.params['batch_size'])):
            return True
        else:
            return False

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if is_training:
            np.random.shuffle(data)
        # Pack until we cannot fit more graphs in the batch
        state_dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        edge_weights_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        intervalGraphIndex = 0
        toLine = self.params['toLine']
        while intervalGraphIndex < len(data):
            num_graphs = 0
            #number of nodes in intervals in the batch
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_converted_node_features = []
            batch_SL = []
            targetLength = []
            batch_target_task_values = []
            batch_target_task_mask = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_num_incoming_edges_per_type = []
            batchNumIncomingEdgesPerType= []
            batch_graph_nodes_list = []
            batch_attention_weight = []
            #number of nodes in the current interval.
            batchNumNodes = []
            batchIntervalGraphNodesList = []
            batchIntervalGraphAdjLists = [[] for _ in range(self.num_edge_types)]
            # node labels
            batchInterIntervalNodeLabels = []
            batchIntraIntervalNodeLabels = []
            batchInterLabelMasks = []
            batchIntraLabelMasks = []
            batchIntraValidLabelMasks = []
            batchGraphMasks = []
            batchInterIntervalNodeLabelsIndex = []
            batchIntraIntervalNodeLabelsIndex = []
            batchInterLabelIndex = []
            batchIntraLabelIndex = []
            batchNumOfValidNodesInGraph = []
            batchFileHash = []
            batchNodeMask = []
            # this is the number of intervals
            batchNumOfNodesInGraph = []
            batchNumOfNodesInGraphPad = [] #[#nodes]
            # total offset
            node_offset = 0
            intervalOffset = 0
            validNodeOffset = 0
            numOfGraphs = 0
            while self.batchCond(intervalGraphIndex, data, num_graphs, node_offset, numOfGraphs):
                intraNodeLabels = []
                intraLabelIndex = []
                numOfNodesInGraph = 0
                numOfValidNodesInGraph = 0
                numOfIntervalInGraph = 0
                for subIndex in range(len(data[intervalGraphIndex])):
                    cur_graph = data[intervalGraphIndex][subIndex]
                    if cur_graph['insideinterval'] == 0:
                        num_nodes_in_graph = cur_graph['numOfNode']
                        numOfIntervalInGraph = num_nodes_in_graph
                        batchNumOfNodesInGraph.append(num_nodes_in_graph)
                        batchIntervalGraphNodesList.append(np.full(shape=[num_nodes_in_graph],
                            fill_value=num_graphs - num_graphs_in_batch, dtype=np.int32))
                        isInterval = False
                        batchInterLabelIndex.append([j for j in range(node_offset-intervalOffset,node_offset-intervalOffset+num_nodes_in_graph)])
                        batchInterIntervalNodeLabels.extend(np.array(cur_graph["bugPos"]))
                        batchInterIntervalNodeLabelsIndex.append(utils.categoryToIndex(cur_graph["bugPos"]))
                    else:
                        # note that, intraNodeLabels is also a list of all nodes in a graph
                        intraNodeLabels += cur_graph["bugPos"]
                        batchIntraIntervalNodeLabels.extend(np.array(cur_graph["bugPos"]))
                        batchNodeMask.extend(np.array(cur_graph["nodeMask"]))
                        num_nodes_in_graph = len(cur_graph['init'])
                        validNumOfNodesInInterval = len(cur_graph['bugPos'])
                        numOfNodesInGraph += num_nodes_in_graph
                        numOfValidNodesInGraph += validNumOfNodesInInterval
                        intraLabelIndex.extend([j for j in range(validNodeOffset, validNodeOffset+validNumOfNodesInInterval)])
                        validNodeOffset += validNumOfNodesInInterval
                        padded_features = np.pad(cur_graph['init'],
                                ((0, 0), (0, self.params['hidden_size'] - self.annotation_size)),
                                'constant')

                        sl = []
                        # use LSTM instead of raw encoding
                        #conv_padded_features = np.zeros([num_nodes_in_graph,self.params['hidden_size']])
                        #conv_padded_features, sl = self.convertRepresentation(padded_features)
                        conv_padded_features = cur_graph['convRep']
                        sl = cur_graph['sl']

                        batch_node_features.extend(padded_features)
                        batch_converted_node_features.extend(conv_padded_features)
                        batch_SL.extend(sl)
                        batchNumNodes.append(np.full(shape=[self.params['hidden_size']],
                            fill_value=num_nodes_in_graph, dtype=np.float32))
                        batch_graph_nodes_list.append(np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                        batchNumOfNodesInGraphPad.append(np.full(shape=[num_nodes_in_graph], fill_value=num_nodes_in_graph, dtype=np.int32))
                        batch_attention_weight.append(np.full(shape=[num_nodes_in_graph, self.params['hidden_size']], fill_value=1.0/num_nodes_in_graph, dtype=np.float32))
                        isInterval = True

                    for i in range(self.num_edge_types):
                        if i in cur_graph['adjacency_lists']:
                            if isInterval:
                                batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + intervalOffset)
                            else:
                                batchIntervalGraphAdjLists[i].append(cur_graph['adjacency_lists'][i] + node_offset - intervalOffset)

                    # Turn counters for incoming edges into np array:
                    num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.num_edge_types))
                    for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                        for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                            if node_id >= num_nodes_in_graph:
                                # happends when some "graph" field from input is empty.
                                print(num_nodes_in_graph)
                                print(num_incoming_edges_per_type_dict)
                            if e_type >= self.num_edge_types:
                                print(num_nodes_in_graph, self.num_edge_types, node_id, e_type)
                                print(num_incoming_edges_per_type_dict)
                            num_incoming_edges_per_type[node_id, e_type] = edge_count
                    if isInterval:
                        batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)
                    else:
                        batchNumIncomingEdgesPerType.append(num_incoming_edges_per_type)

                    target_task_values = []
                    target_task_mask = []
                    if not isInterval:
                        for target_val in cur_graph['labels']:
                            batch_target_task_values.append(target_val)
                            batch_target_task_mask.append([1.])
                            targetLength.append(len(target_val))

                        batchFileHash.append(cur_graph['fileHash'])

                    num_graphs += 1
                    node_offset += num_nodes_in_graph
                    if isInterval:
                        intervalOffset += num_nodes_in_graph
                        num_graphs_in_batch += 1
                numOfGraphs += 1
                batchIntraLabelIndex.append(intraLabelIndex)
                batchNumOfValidNodesInGraph.append(numOfValidNodesInGraph)
                batchIntraIntervalNodeLabelsIndex.append(utils.categoryToIndex(intraNodeLabels))
                isBuggy = self.isBuggyMethod(data[intervalGraphIndex])
                batchIntraValidLabelMasks.extend([isBuggy for i in range(numOfValidNodesInGraph)])
                batchIntraLabelMasks.extend([isBuggy for i in range(numOfNodesInGraph)])
                batchInterLabelMasks.extend([isBuggy for i in range(numOfIntervalInGraph)])
                batchGraphMasks.append(isBuggy)
                intervalGraphIndex += 1

            if len(batchIntraIntervalNodeLabels) == 0:
                #empty nodes
                continue

            batchInterLabelIndex = np.array(list(itertools.zip_longest(*batchInterLabelIndex, fillvalue=node_offset-intervalOffset))).T
            #g[self.placeholders['intraLabelIndex']]: [[1,2,3,10], [4,5,10,10], [6,7,8,9]]
            batchIntraLabelIndex = np.array(list(itertools.zip_longest(*batchIntraLabelIndex, fillvalue=validNodeOffset))).T
            if len(batch_target_task_values[0]) == 1:
                batch_target_task_values = np.transpose(batch_target_task_values, axes=[1,0])
            else:
                batch_target_task_values.append([0 for i in range(self.params["maxTargetLength"])])
                batch_target_task_values = np.array(list(itertools.zip_longest(*batch_target_task_values, fillvalue=1))).T
                batch_target_task_values = batch_target_task_values[:-1]
            batch_target_task_mask = np.transpose(batch_target_task_mask, axes=[1,0])


            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['converted_node_representation']: np.array(batch_converted_node_features),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type, axis=0),
                # list of nodes inside intervals. [0,0,1,1,1,1]
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                self.placeholders['attentionWeight']: np.concatenate(batch_attention_weight),
                self.placeholders['target_values']: batch_target_task_values,
                self.placeholders['target_mask']: batch_target_task_mask,
                self.placeholders['num_graphs']: numOfGraphs,
                self.placeholders['numOfNodesInGraph']: np.array(batchNumOfNodesInGraph),
                self.placeholders['numOfNodesInGraphsPad']: np.concatenate(batchNumOfNodesInGraphPad),
                self.placeholders['graph_state_keep_prob']: state_dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: edge_weights_dropout_keep_prob,

                self.placeholders['batch_SL']: np.array(batch_SL),
                self.placeholders['targetLength']: np.array(targetLength),

                self.placeholders['interIntervalNodeLabels']: np.array(batchInterIntervalNodeLabels),
                self.placeholders['intraIntervalNodeLabels']: np.array(batchIntraIntervalNodeLabels),
                self.placeholders['interIntervalNodeLabelsIndex']: np.array(batchInterIntervalNodeLabelsIndex),
                self.placeholders['intraIntervalNodeLabelsIndex']: np.array(batchIntraIntervalNodeLabelsIndex),
                # number of nodes in each interals. [2,3,2]
                self.placeholders['intervalGraphNodesList']: np.concatenate(batchIntervalGraphNodesList),
                self.placeholders['numIntervalGraphs']: num_graphs - num_graphs_in_batch,
                self.placeholders['numIntraGraphs']: num_graphs_in_batch,
                self.placeholders['numOfNodeInSubgraph']: np.array(batchNumNodes),
                self.placeholders['intervalNumIncomingEdgesPerType']: np.concatenate(batchNumIncomingEdgesPerType),
                # this is the number of all nodes (statements) in a CFG.
                self.placeholders['numOfValidNodesInGraph']: batchNumOfValidNodesInGraph,
                self.placeholders['fileHash']: batchFileHash,
                self.placeholders['nodeMask']: batchNodeMask,
                self.placeholders['intraLabelMasks']: batchIntraLabelMasks,
                self.placeholders['intraValidLabelMasks']: batchIntraValidLabelMasks,
                self.placeholders['interLabelMasks']: batchInterLabelMasks,
                self.placeholders['graphLabelMasks']: batchGraphMasks,
                self.placeholders['interLabelIndex']: batchInterLabelIndex,
                self.placeholders['intraLabelIndex']: batchIntraLabelIndex
            }

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

                if len(batchIntervalGraphAdjLists[i]) > 0:
                    intAdjList = np.concatenate(batchIntervalGraphAdjLists[i])
                else:
                    intAdjList = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['intervalAdjLists'][i]] = intAdjList

            yield batch_feed_dict


def main():
    args = docopt(__doc__)
    try:
        model = IntervalGGNNModel(args)
        model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
