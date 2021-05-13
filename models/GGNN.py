#!/usr/bin/env/python
"""
Usage:
    GGNN.py [options]

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
import itertools
import sys, traceback
import pdb

from GNN_base import BaseModel
from utils import glorot_init, SMALL_NUMBER
import utils



GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells',])


class SparseGGNNModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'batch_size': 100000,
            'use_edge_bias': False,
            'maxTargetLength': -1,
            'nodesThreshold': 1000,
            'use_propagation_attention': False,
            'use_edge_msg_avg_aggregation': True,
            'residual_connections': {  # For layer i, specify list of layers whose output is added as an input
                                     "2": [0],
                                     "4": [0, 2]
                                    },

            "maxWidth": 1,#only for sandwiches-sparse
            'layer_timesteps': [2, 2, 1, 2, 1],  # number of layers & propagation steps per layer

            'graph_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell, or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'outputCSVPrefix': "",
            'filterLabel': 0,
            'task_sample_ratios': {},
            'edge_weight_dropout_keep_prob': .8
        })
        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, h_dim],
                                                                          name='node_features')
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.num_edge_types)]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.num_edge_types],
                                                                          name='num_incoming_edges_per_type')
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='graph_nodes_list')
        self.placeholders['fileHash'] = tf.placeholder(tf.string, [None], name='fileHash')
        self.placeholders['nodeMask'] = tf.placeholder(tf.bool, [None], name='nodeMask')
        self.placeholders['nodeLabels'] = tf.placeholder(tf.int32, [None], name='nodeLabels')
        self.placeholders['nodeLabelIndex'] = tf.placeholder(tf.int32, [None], name='nodeLabelIndex')
        self.placeholders['numOfNodesInGraph'] = tf.placeholder(tf.int32, [None], name='numOfNodesInGraph')
        self.placeholders['labelIndex'] = tf.placeholder(tf.int32, [None, None], name='labelIndex')
        self.placeholders['labelMasks'] = tf.placeholder(tf.int32, [None], name='labelMasks')
        self.placeholders['graphLabelMasks'] = tf.placeholder(tf.int32, [None], name='graphLabelMasks')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')
        self.placeholders['batch_SL'] = tf.placeholder(tf.int32, [None], name='batch_SL')
        self.placeholders['targetLength'] = tf.placeholder(tf.int32, [None], name='targetLength')
        self.placeholders['converted_node_representation'] = tf.placeholder(tf.int32,
                [None, self.params["maxWidth"]],
                name='convertedNodeRep')

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


    def compute_final_node_representations(self) -> tf.Tensor:
        #self.convertRepresentation()

        node_states_per_layer = []  # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer.append(self.placeholders['initial_node_representation'])
        num_nodes = tf.shape(self.placeholders['initial_node_representation'], out_type=tf.int32)[0]

        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_types = []  # list of tensors of edge type of shape [E]
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['adjacency_lists']):
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
                        for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['adjacency_lists']):
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
                            incoming_messages += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
                                                           self.gnn_weights.edge_biases[layer_idx])  # Shape [V, D]

                        if self.params['use_edge_msg_avg_aggregation']:
                            num_incoming_edges = tf.reduce_sum(self.placeholders['num_incoming_edges_per_type'],
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

        # Sum up all nodes per-graph
        graph_representations = tf.unsorted_segment_sum(data=gated_outputs,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g x 1]
        return tf.squeeze(graph_representations)  # [g]

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        processed_graphs = []
        self.number_of_tokens = 0
        totalNumOfWholeGraphs = 0
        totalNumOfNodes = 0
        numOfNodesDropped = 0
        numNodesList = []
        self.ops['maxTargetLen'] = 0

        # this is used to drop complex CFGs.
        nodesThreshold = self.params["nodesThreshold"]
        if self.params["on_large_data"] != 100:
            nodesThreshold = 100000

        for d in raw_data:
            (adjacency_lists, num_incoming_edge_per_type) = self.__graph_to_adjacency_lists(d['graph'])
            numOfNodes = len(d["node_features"])

            if numOfNodes > nodesThreshold:
                numOfNodesDropped += 1
                continue
            if len(d["bugPos"]) == 0:
                #TODO: this should not happen
                continue

            labels = [d["targets"][task_id] for task_id in self.params['task_ids']]
            if len(labels) > 0 and len(labels[0]) > self.ops["maxTargetLen"]:
                self.ops["maxTargetLen"] = len(labels[0])
            if d.get("convRep") == None:
                d["convRep"] = [[0] for i in range(numOfNodes)]
            if d.get("numOfFeatures") == None:
                d["numOfFeatures"] = [len(d["convRep"][i]) for i in range(len(d["convRep"]))]
            processed_graphs.append({"adjacency_lists": adjacency_lists,
                                     "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                     "init": d["node_features"],
                                     "bugPos": d["bugPos"],
                                     "nodeMask": d["node_mask"],
                                     "fileHash": str(d["fileHash"][0]) + "-"+d['funName'],
                                     "projName": d["projName"],
                                     "convRep": d["convRep"],
                                     "sl":d["numOfFeatures"],
                                     "labels": labels})
            if len(d["node_features"][0]) > self.number_of_tokens:
                self.number_of_tokens = len(d["node_features"][0])
            totalNumOfNodes += len(d['node_features'])
            numNodesList.append(len(d['node_features']))
            totalNumOfWholeGraphs += 1

        print("total graph: %d , dropped: %d and aveNodes: %f" %
                (totalNumOfWholeGraphs, numOfNodesDropped, float(totalNumOfNodes)/totalNumOfWholeGraphs))
        if is_training_data:
            np.random.shuffle(processed_graphs)
            for task_id in self.params['task_ids']:
                task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                if task_sample_ratio is not None:
                    ex_to_sample = int(len(processed_graphs) * task_sample_ratio)
                    for ex_id in range(ex_to_sample, len(processed_graphs)):
                        processed_graphs[ex_id]['labels'][task_id] = None

        #TODO: added to eval performance on a specific percent of data
        if is_training_data:
            utils.filterGraphByPerc(processed_graphs, self.params['on_large_data'])
            print("after filtering, remain %d graphs. min/max is %d/%d "%(len(processed_graphs), min(numNodesList), max(numNodesList)))
        return processed_graphs

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

    def convertRepresentation(self, rep):
        #[[1,1,2,1], [0, 0, 1, 1]] => [[0,1,2,2,3], [2, 3]], [5, 2]
        newRep = []
        sl = []
        for r in rep:
            newR = []
            for index in range(len(r)):
                newR.extend([index]*r[index])
            sl.append(len(newR))
            newRep.append(newR)
        return newRep, sl

    def isBuggyMethod(self, g):
        if g["labels"][0] == 0.0:
            return False
        if 1 in g["bugPos"]:
            return True
        return False

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if is_training:
            np.random.shuffle(data)
        # Pack until we cannot fit more graphs in the batch
        state_dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        edge_weights_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_converted_node_features = []
            batch_SL = []
            targetLength = []
            batch_target_task_values = []
            batch_target_task_mask = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0
            batchFileHash = []
            batchNodeMask = []
            batchBugPos = []
            batchBugPosIndex = []
            batchNumOfNodesInGraph = []
            batchLabelMasks = []
            batchGraphMasks = []
            batchLabelIndex = []

            while num_graphs < len(data) and (num_graphs_in_batch == 0 or (node_offset + len(data[num_graphs]['init']) < self.params['batch_size'])):
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph['init'])
                padded_features = np.pad(cur_graph['init'],
                                         ((0, 0), (0, self.params['hidden_size'] - self.annotation_size)),
                                         'constant')
                sl = []
                # use LSTM instead of raw encoding
                #padded_features, sl = self.convertRepresentation(padded_features)

                batch_node_features.extend(padded_features)
                sl = cur_graph["sl"]
                batch_SL.extend(sl)
                batch_graph_nodes_list.append(np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                for i in range(self.num_edge_types):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.num_edge_types))
                for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                for target_val in cur_graph['labels']:
                    batch_target_task_values.append(target_val)
                    batch_target_task_mask.append([1.])
                    targetLength.append(len(target_val))
                batchFileHash.append(cur_graph["fileHash"])
                batchBugPos.extend(np.array(cur_graph["bugPos"]))
                batchNodeMask.extend(np.array(cur_graph["nodeMask"]))
                batchBugPosIndex.append(utils.categoryToIndex(cur_graph["bugPos"]))
                isBuggy = self.isBuggyMethod(data[num_graphs])
                batchLabelMasks.extend([isBuggy for i in range(num_nodes_in_graph)])
                batchGraphMasks.append(isBuggy)
                totalBugPosNum = sum(batchNumOfNodesInGraph)
                labelIndex = [j for j in range(totalBugPosNum, totalBugPosNum + len(cur_graph["bugPos"]))]
                batchLabelIndex.append(labelIndex)
                batchNumOfNodesInGraph.append(len(cur_graph["bugPos"]))
                conv_padded_features = cur_graph['convRep']
                sl = cur_graph['sl']
                batch_converted_node_features.extend(conv_padded_features)

                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            if len(batch_target_task_values[0]) == 1:
                batch_target_task_values = np.transpose(batch_target_task_values, axes=[1,0])
            else:
                batch_target_task_values.append([0 for i in range(self.params["maxTargetLength"])])
                batch_target_task_values = np.array(list(itertools.zip_longest(*batch_target_task_values, fillvalue=1))).T
                batch_target_task_values = batch_target_task_values[:-1]


            batchLabelIndex = np.array(list(itertools.zip_longest(*batchLabelIndex,
                fillvalue=sum(batchNumOfNodesInGraph)))).T
            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type, axis=0),
                self.placeholders['converted_node_representation']: np.array(batch_converted_node_features),
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                self.placeholders['target_values']: batch_target_task_values,
                self.placeholders['fileHash']: batchFileHash,
                self.placeholders['nodeMask']: batchNodeMask,
                self.placeholders['nodeLabels']: np.array(batchBugPos),
                self.placeholders['nodeLabelIndex']: batchBugPosIndex,
                self.placeholders['labelIndex']: batchLabelIndex,
                self.placeholders['target_mask']: np.transpose(batch_target_task_mask, axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['numOfNodesInGraph']: batchNumOfNodesInGraph,
                self.placeholders['labelMasks']: batchLabelMasks,
                self.placeholders['graphLabelMasks']: batchGraphMasks,
                self.placeholders['graph_state_keep_prob']: state_dropout_keep_prob,
                self.placeholders['batch_SL']: np.array(batch_SL),
                self.placeholders['targetLength']: np.array(targetLength),
                self.placeholders['edge_weight_dropout_keep_prob']: edge_weights_dropout_keep_prob
            }

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict

def main():
    args = docopt(__doc__)
    try:
        model = SparseGGNNModel(args)
        model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
