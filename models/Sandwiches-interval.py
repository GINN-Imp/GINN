#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_sparse.py [options]

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

from GINN import IntervalGGNNModel
from utils import MLP, glorot_init, SMALL_NUMBER, computeTopN
import utils

from tensor2tensor.models import transformer, lstm
from tensor2tensor.data_generators import problem_hparams



GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells',])


class SandwichesInterval(IntervalGGNNModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            #'batch_size': 100000,
            'batch_size': 200,
            'compMisuse': -1,
            'batch_graph_size': 4,
            'rep_mode': 0,
            'toLine': False,
            #'train_file': 'intervals-jfreechart-1.0.19.json',
            'train_file': '/proj/fff000/ggnnbl/spoon-intervals/jsondata/intervals-projects-defects4j-train.json',
            'valid_file': '/proj/fff000/ggnnbl/spoon-intervals/jsondata/intervals-projects-defects4j-train.json',
            'use_edge_bias': False,
            'accInfoPreprocess': False,
            'debugAccInfo': False,
            'threeEmbedding': False,
            'filterLabel': 0,
            'outputCSVPrefix': "",
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
            'filterTrain': False,
            'vocab_dir': "/Users/fff000/Documents/research/GGNNBL/GGNN-data/scripts/structured-neural-summarization/data/java-data/vocabInfo.json",
            'edge_weight_dropout_keep_prob': .8
        })
        return params

    '''
    def batchCond(self, intervalGraphIndex, data, num_graphs, node_offset, numOfGraphs):
        if intervalGraphIndex < len(data) and (numOfGraphs == 0 or (numOfGraphs < self.params['batch_graph_size'])):
            return True
        else:
            return False
    '''

    def load_data(self, file_name, is_training_data: bool):
        processed_graphs = IntervalGGNNModel.load_data(self, file_name, is_training_data)
        vocabDir = self.params['vocab_dir']
        with open(vocabDir) as f:
            p = json.load(f)
            # includes vocab size (ie., output), max sentence length (ie., length)
            self.ops['vocabInfo'] = p
            self.number_of_tokens = self.ops['vocabInfo']["node"]
        return processed_graphs

    def prepare_specific_graph_model(self) -> None:
        super().prepare_specific_graph_model()
        self.placeholders['gatherNodeIndice'] = tf.placeholder(tf.int32, [None, 2],
                name='gatherNodeIndice')
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [None, None],
                name='target_values')
        self.placeholders['misuse_pos'] = tf.placeholder(tf.int32, [None],
                name='misuse_pos')
        self.placeholders['repair_pos'] = tf.placeholder(tf.int32, [None],
                name='repair_pos')
        self.placeholders['merge_pos'] = tf.placeholder(tf.int32, [2,None],
                name='merge_pos')
        self.placeholders['nodeIndexInGraph'] = tf.placeholder(tf.int32, [None, 1],
                name='nodeIndexInGraph')
        self.placeholders['numTokenInGraph'] = tf.placeholder(tf.int32, [None],
                name='numTokenInGraph')
        self.placeholders['nodeIndex'] = tf.placeholder(tf.int32, [None, None],
                name='nodeIndex')

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        for g in super().make_minibatch_iterator(data, is_training):
            numOfValidNodesInGraph = g[self.placeholders['numOfValidNodesInGraph']]
            nodeMask = g[self.placeholders['nodeMask']]
            target_values = g[self.placeholders['target_values']]
            gatherNodeIndice = []
            nodeIndexInGraph = []
            misuse_pos = []
            repair_pos = []
            for i in range(len(numOfValidNodesInGraph)):
                numOfValidNodes = numOfValidNodesInGraph[i]
                target_value = target_values[i]
                misPos = [0 for i in range(numOfValidNodes)]
                misPos[int(target_value[1])] = 1
                misuse_pos += misPos
                repPos = [0 for i in range(numOfValidNodes)]
                repPos[int(target_value[2])] = 1
                repair_pos += repPos
                for j in range(numOfValidNodes):
                    gatherNodeIndice.append([i,j])
                    nodeIndexInGraph.append([i])
            g[self.placeholders["gatherNodeIndice"]] = np.array(gatherNodeIndice)
            #print(nodeIndexInGraph)
            g[self.placeholders["nodeIndexInGraph"]] = np.array(nodeIndexInGraph)
            misuse_pos = utils.doMask(misuse_pos, nodeMask)
            repair_pos = utils.doMask(repair_pos, nodeMask)
            tokens = utils.doMask(nodeIndexInGraph, nodeMask)
            tokens = [i[0] for i in tokens]
            numTokenInGraph = utils.countNodes(tokens)
            nodeIndex = utils.nodeCount2NodeIndex(numTokenInGraph)
            g[self.placeholders["nodeIndex"]] = nodeIndex
            g[self.placeholders["numTokenInGraph"]] = np.array(numTokenInGraph)
            g[self.placeholders["misuse_pos"]] = np.array(misuse_pos)
            g[self.placeholders["repair_pos"]] = np.array(repair_pos)
            g[self.placeholders["merge_pos"]] = np.stack([misuse_pos, repair_pos])
            #print(g[self.placeholders['target_values']])
            yield g

    def useRNN(self):
        hidden_size = self.params["hidden_size"]
        #self.placeholders['intraLabelIndex']: [[1,2,3,10], [4,5,10,10], [6,7,8,9]]
        intraLabelIndex = self.placeholders['intraLabelIndex'] # [3,4]
        #self.placeholders['numOfValidNodesInGraph']: [3,2,4]
        numOfValidNodesInGraph = self.placeholders['numOfValidNodesInGraph']
        #converted_node_representation: [[1],[2]], assume the shape is [None, 1]
        x = self.placeholders['converted_node_representation']
        self.embeddings = tf.get_variable('embedding_matrix', [self.number_of_tokens, self.params["hidden_size"]])
        gatherNodeIndice = self.placeholders["gatherNodeIndice"]
        initNR = utils.useRNN(hidden_size, intraLabelIndex, numOfValidNodesInGraph, x, self.embeddings, gatherNodeIndice)
        self.placeholders['initial_node_representation'] = initNR

    def stateRNN(self, final_node_representations, name):
        hidden_size = self.params["hidden_size"]
        #self.placeholders['intraLabelIndex']: [[1,2,3,10], [4,5,10,10], [6,7,8,9]]
        intraLabelIndex = self.placeholders['intraLabelIndex']
        #self.placeholders['numOfValidNodesInGraph']: [3,2,4]
        numOfValidNodesInGraph = self.placeholders['numOfValidNodesInGraph']
        #converted_node_representation: [[1],[2]], assume the shape is [None, 1]
        gatherNodeIndice = self.placeholders["gatherNodeIndice"]
        iniNP, self.finalStat = utils.stateRNN(final_node_representations, hidden_size, intraLabelIndex, numOfValidNodesInGraph, gatherNodeIndice, name)
        return iniNP
        #self.placeholders['initial_node_representation'] = iniNP

    def convertInitialNodeRep(self):
        mask = self.placeholders['nodeMask']
        revMask = tf.math.logical_not(mask)
        x = self.placeholders['converted_node_representation']
        self.embeddings = tf.get_variable('embedding_matrix', [self.number_of_tokens, self.params["hidden_size"]])
        sl = self.placeholders['batch_SL']
        initial_node_representation = self.placeholders['initial_node_representation']
        self.placeholders['initial_node_representation'] = utils.convertInitialNodeRep(mask, x, self.embeddings, sl, initial_node_representation)

    def loopRNN(self, final_node_representations, name):
        mask = self.placeholders['nodeMask']
        x = self.placeholders['converted_node_representation']
        sl = self.placeholders['batch_SL']
        self.placeholders['initial_node_representation'] = utils.loopRNN(mask, x, sl, self.embeddings, final_node_representations, name)

    def mode1(self):
        self.useRNN()
        final_node_representations = self.compute_final_node_representations_inside()
        final = self.stateRNN(final_node_representations, "loop1")
        self.placeholders['initial_node_representation'] = final
        return super().compute_final_node_representations()

    def mode2(self):
        self.convertInitialNodeRep()
        final_node_representations = self.compute_final_node_representations_inside()
        self.loopRNN(final_node_representations, "loop1")
        return self.compute_final_node_representations_inside()

    def compute_final_node_representations(self):
        res = self.mode1()
        return res

    def computeLoss1(self, task_id, internal_id):
        if self.params["use_attention"]:
            self.calAttention()
            self.computeAttentionWeight()
        pred = self.computePred(task_id, internal_id)
        y = tf.to_int64(self.placeholders['target_values'])
        y = tf.slice(y, [0, 0], [-1, 1]) # [None, 1]
        y = tf.squeeze(y, axis=1)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y)
        pred = tf.argmax(pred,1)
        correct_pred = tf.equal(pred, y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #self.ops['pred'] = tf.argmax(pred,1)
        self.ops['accuracy_task%i' % task_id] = accuracy
        self.ops['losses'].append(cost)
        self.ops['acc_info'] = [pred, y, self.placeholders['numTokenInGraph'], self.placeholders['numOfValidNodesInGraph']]

    def computeLoc(self, tokenRep, finalRep):
        name = "colsplit"
        hidden_size = self.params['hidden_size']
        with tf.variable_scope(name):
            with tf.variable_scope("W1"):
                W1 = MLP(hidden_size, hidden_size, [], 1.0)
            with tf.variable_scope("W2"):
                W2 = MLP(hidden_size, hidden_size, [], 1.0)
            with tf.variable_scope("W3"):
                W3 = MLP(hidden_size, 2, [], 1.0)

            mask = self.placeholders['nodeMask']
            nodeIndexInGraph = self.placeholders['nodeIndexInGraph'] #[#Nodes,1]
            H2 = tf.gather_nd(finalRep, nodeIndexInGraph) #[#Nodes, 100]
            H2 = tf.boolean_mask(H2, mask)

            H1 = tf.boolean_mask(tokenRep, mask)

            E1 = W1(H1) + W2(H2) # [#Nodes, 100]
            E2 = W3(E1) #[#Nodes, 2]
            newE2 = tf.transpose(E2) #[2, #Nodes]
            return newE2
            #loc = tf.slice(E2, [0, 0], [-1, 1]) # [None, 1]
            #repair = tf.slice(E2, [0, 1], [-1, 1]) # [None, 1]
            #return loc, repair

    def computeLoss2(self, task_id, internal_id):
        if self.params["use_attention"]:
            self.calAttention()
            self.computeAttentionWeight()
        finalRep = self.ops['final_node_representations']
        #predLoc, predRep = self.computeLoc(finalRep, self.finalStat)
        #pred = self.computeLoc(finalRep, self.finalStat)
        pred = utils.computeLoc(self, finalRep, self.finalStat)
        #y = tf.to_int64(self.placeholders['target_values'])
        # the merge_pos and related pos is already go throught node_mask.
        merge_pos = self.placeholders['merge_pos']
        pred = tf.nn.softmax(pred)
        #misuse_pos = tf.to_int64(self.placeholders['misuse_pos'])
        #repair_pos = tf.to_int64(self.placeholders['repair_pos'])

        #cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=locRep)
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=merge_pos)
        tmpPred = tf.argmax(pred,1)
        tmpY = tf.argmax(merge_pos,1)
        correct_pred = tf.equal(tmpPred, tmpY)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #self.ops['pred'] = tf.argmax(pred,1)
        self.ops['accuracy_task%i' % task_id] = accuracy
        self.ops['losses'].append(cost)
        self.ops['acc_info'] = [pred, merge_pos, self.placeholders['numTokenInGraph']]

    def computeLoss(self, task_id, internal_id):
        if self.params['toLine'] == False:
            self.computeLoss1(task_id, internal_id)
        else:
            utils.computeLoss3(self, task_id, internal_id, self.params["compMisuse"])
            #self.computeLoss2(task_id, internal_id)

    def parseAccInfo(self, best_accInfo):
        return
        if self.params['toLine'] == False:
            super().parseAccInfo(best_accInfo)
            return
        pred = best_accInfo[0]
        merge_pos = best_accInfo[1]
        nodeIndexInGraph = best_accInfo[2].tolist()
        acc = [0,0]
        '''
        for i in range(2):
            colPred = pred[0].tolist()
            colTar = merge_pos[0].tolist()
            acc[i] = utils.computeTopNBySeq(colPred, colTar, nodeIndexInGraph, 1)
        acc.append(acc[0]*acc[1])
        print("  avg node is:%.1f, loc acc is: %.3f, repair acc is: %.3f, combine acc is: %.3f." % (sum(nodeIndexInGraph) / len(nodeIndexInGraph), acc[0], acc[1], acc[2]))
        '''
        compMisuse = self.params["compMisuse"]
        pred0 = pred[0].tolist()
        merge_pos0 = merge_pos[0].tolist()
        pred1 = pred[1].tolist()
        merge_pos1 = merge_pos[1].tolist()
        utils.computeCombineBySeq(pred0, merge_pos0, pred1, merge_pos1, nodeIndexInGraph, 1)
        if compMisuse == 0:
            utils.dumpRes((pred1, merge_pos1, nodeIndexInGraph), "compMisuse0.pkl")
        elif compMisuse == 1:
            utils.dumpRes((pred0, merge_pos0, nodeIndexInGraph), "compMisuse1.pkl")

def main():
    args = docopt(__doc__)
    try:
        model = SandwichesInterval(args)
        model.train()
    except:
        print()
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
