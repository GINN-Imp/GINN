#!/usr/bin/env/python

import itertools
import numpy as np
import tensorflow as tf
import queue
import threading

SMALL_NUMBER = 1e-7


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int=2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()

def pickResBaseOnFile(graphPred, y, graphFileHash):
    res = {}
    for index in range(len(graphFileHash)):
        fileHash = graphFileHash[index]
        if res.get(fileHash) == None:
            res[fileHash] = [[], []]
        res[fileHash][0].append(graphPred[index])
        res[fileHash][1].append(y[index])
    return res

def returnTopNRes(preds, indices, n, ifkeep=False):
    pos = 0
    res = []
    for i in range(len(indices)):
        labelInd = indices[i]
        predRes = []
        for predIndex in range(pos, pos+labelInd):
            predRes.append(preds[predIndex])
        pos += labelInd
        sortedIndex = sorted(range(len(predRes)), key=lambda k:predRes[k], reverse = True)
        predRes = sortedIndex[:n]
        if ifkeep:
            resArr = [0 for j in range(labelInd)]
            for j in predRes:
                resArr[j] = 1
            predRes = resArr
        res.append(predRes)
    return res

def returnResByThreshold(preds, indices, threshold):
    # pred = [.1,.2,.6,.1,.2,.6,.7] indices=[3,4] n=0.5
    # [[0, 0, 1], [0, 0, 1, 1]]
    pos = 0
    res = []
    for i in range(len(indices)):
        labelInd = indices[i]
        predRes = []
        for predIndex in range(pos, pos+labelInd):
            if preds[predIndex] >= threshold:
                predRes.append(1)
            else:
                predRes.append(0)
        res.append(predRes)
        pos += labelInd
    return res

def returnSEMet(res):
    fgp = res[0]
    fy = res[1]
    fgp = np.array(fgp)
    fy = np.array(fy)
    TP = np.count_nonzero(fgp * fy)
    TN = np.count_nonzero((fgp - 1) * (fy - 1))
    FP = np.count_nonzero(fgp * (fy - 1))
    FN = np.count_nonzero((fgp - 1) * fy)
    return TP, TN, FP, FN

def returnSEMetByPortion(pred, y, numOfNodes, stride = 10):
    if stride == 0:
        return
    sortedIndex = sorted(range(len(numOfNodes)), key=lambda k:numOfNodes[k])
    indicesList = []
    for i in range(int(100/stride)):
        begin = int(len(numOfNodes) * i * stride / 100)
        end = int(len(numOfNodes) * (i + 1) * stride / 100)
        indicesList.append(sortedIndex[begin:end])
    for indice in range(len(indicesList)):
        newPred = []
        newY = []
        for i in indicesList[indice]:
            newPred.append(pred[i])
            newY.append(y[i])

        TP, TN, FP, FN = returnSEMet((newPred, newY))
        acc = float(TP+TN)/ (TP+TN+FP+FN)
        print(str(indice*stride) + " to " + str(indice*stride+stride) +": acc is: " + str(round(acc, 3)), end="")
        computeF1(TP, TN, FP, FN)


def writeToCSV(info, filename):
    import csv
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fileHash", "TP", "TN", "FP", "FN"])
        for i in info:
            writer.writerow([str(e) for e in i])

def computeF1(TP, TN, FP, FN):
    if TP + FP == 0:
        precision = -2.0
    else:
        precision = float(TP) / (TP + FP)
    if TP + FN == 0:
        recall = -2.0
    else:
        recall = float(TP) / (TP + FN)
    acc = float(TP+TN)/ (TP+TN+FP+FN)
    if precision + recall == 0:
        f1 = -2.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    print("  (precision is: %.5f, recall is: %.5f, f1 is: %.5f)" %
            (precision, recall, f1))

def constructLabelByIndices(y, indices):
    res = []
    for i in range(len(y)):
        numOfNodes = indices[i]
        tmp = [0 for j in range(numOfNodes)]
        tmp[y[i]] = 1
        res.append(tmp)
    return np.concatenate(res)

def pickResBaseOnTopN(graphPred, y, graphFileHash, indices):
    res = {}
    pos = 0
    for index in range(len(graphFileHash)):
        fileHash = graphFileHash[index]
        numOfNodes = indices[index]
        if res.get(fileHash) == None:
            res[fileHash] = [np.array([], np.int32), np.array([], np.int32)]
        end = pos+numOfNodes
        res[fileHash][0] = np.concatenate((res[fileHash][0], graphPred[pos:end]))
        res[fileHash][1] = np.concatenate((res[fileHash][1], y[pos:end]))
        pos = end
    return res

def writeLineInfo2CSV(res, nodeLabels, buggyFileHash, topN, trainFile, buggyLineInfo, indices):
    res = pickResBaseOnTopN(res, nodeLabels, buggyFileHash, indices)
    for k in res:
        TP, TN, FP, FN = returnSEMet(res[k])
        buggyLineInfo.append([k, TP, TN, FP, FN, topN])

def computeTop1F1(info, filterLabel):
    if filterLabel == 0:
        return 0
    pred = info[0]
    indices = info[1]
    intraNodeLabel = info[2]
    nodeLabels = constructLabelByIndices(intraNodeLabel, indices)
    res = returnTopNRes(pred, indices, 1, True)
    res = np.concatenate(res, axis=0)
    TP, TN, FP, FN = returnSEMet([res, nodeLabels])
    return (TP, TN, FP, FN)

def computeSEMetric(info, trainFile, filterLabel):
    pred = info[0]
    indices = info[1]
    intraNodeLabel = info[2]
    buggyFileHash = info[3]
    graphPred = info[4]
    y = info[5]
    graphFileHash = info[6]
    #compute clean/buggy prediction results for file level.
    total=[0,0,0,0]
    if filterLabel == 0:
        res = pickResBaseOnFile(graphPred, y, graphFileHash)
        methodInfo = []
        for k in res:
            TP, TN, FP, FN = returnSEMet(res[k])
            total[0] += TP
            total[1] += TN
            total[2] += FP
            total[3] += FN
            methodInfo.append([k, TP, TN, FP, FN])
        if trainFile != "":
            filenameMethodInfo = trainFile
            filenameMethodInfo += "-SEMethod.csv"
            writeToCSV(methodInfo, filenameMethodInfo)
        computeF1(total[0], total[1], total[2], total[3])
    #compute buggy lines.
    else:
        buggyLineInfo = []
        # pred = [1,2,3,1,2,3,4] indices=[3,4] n=1 True
        # [[0, 0, 1], [0, 0, 0, 1]]
        nodeLabels = constructLabelByIndices(intraNodeLabel, indices)
        for i in [1,3,5,7,10]:
            res = returnTopNRes(pred, indices, i, True)
            res = np.concatenate(res, axis=0)
            TP, TN, FP, FN = returnSEMet([res, nodeLabels])
            print("top-%d: " % (i))
            computeF1(TP, TN, FP, FN)
            writeLineInfo2CSV(res, nodeLabels, buggyFileHash, i, trainFile, buggyLineInfo, indices)
        if trainFile != "":
            filenameBuggylineInfo = trainFile
            filenameBuggylineInfo += "-SELine.csv"
            writeToCSV(buggyLineInfo, filenameBuggylineInfo)
        res = returnResByThreshold(pred,indices, 0.5)
        res = np.concatenate(res, axis=0)
        TP, TN, FP, FN = returnSEMet([res, nodeLabels])
        print("threshold-based prediction: " )
        computeF1(TP, TN, FP, FN)

def computeTopNAcc(pred, intraNodeLabel, n):
    #res = tf.metrics.mean(tf.nn.in_top_k(predictions=pred, targets=intraNodeLabel, k=n))
    #return res[0]

    _,i = tf.math.top_k(pred, n)
    labels_tiled = tf.tile(tf.expand_dims(intraNodeLabel, axis=-1), [1,n])
    equality = tf.equal(i, labels_tiled)
    logic_or = tf.reduce_any(equality, axis=-1)
    accN = tf.reduce_mean(tf.cast(logic_or, tf.float32))
    return accN

def printTopNAcc(best_accInfo):
    print("  top1 acc is: %.5f, top3 acc is: %.5f, top5 acc is: %.5f, top7 acc is: %.5f, top10 acc is: %.5f." %
            (best_accInfo[0], best_accInfo[1], best_accInfo[2], best_accInfo[3], best_accInfo[4]))

def concatAccInfo(pre, new, num_graphs):
    newRes = []
    if pre == None:
        pre = [[] for i in range(len(new))]
    for i in range(len(pre)):
        newRes.append(list(pre[i])+list(new[i]))
    return newRes


def categoryToIndex(categ):
    #assert len(categ) > 0, "bugpos is empty."
    if len(categ) == 0:
        return 0
    for i in range(len(categ)):
        if categ[i] != 0:
            return i
    return -1

def returnTopNResByThre(preds, indices, threshold=0.0, ifkeep=False):
    pos = 0
    res = []
    for i in range(len(indices)):
        labelInd = indices[i]
        predRes = []
        for predIndex in range(pos, pos+labelInd):
            predRes.append(preds[predIndex])
        pos += labelInd
        sortedIndex = sorted(range(len(predRes)), key=lambda k:predRes[k], reverse = True)
        maxIndex = 0
        for i in range(len(predRes)):
            if predRes[i] > predRes[maxIndex]:
                maxIndex = i
            if predRes[i] > threshold:
                predRes.append(i)
        if len(predRes) == 0:
            predRes.append(maxIndex)

        if ifkeep:
            resArr = [0 for j in range(labelInd)]
            for j in predRes:
                resArr[j] = 1
            predRes = resArr
        res.append(predRes)
    return res

def computeTopNByThre(preds, indices, labels, threshold):
    # preds: [0,1,0,0,1,0,0,1], indices: [2,3,3], labels: [0,2,1]
    res = [0,0]
    pos = 0
    indLength = 0
    for i in indices:
        indLength += i
    assert indLength == len(preds), "Inconsist length"

    predRes = returnTopNResByThre(preds, indices, threshold)

    assert len(predRes) == len(labels), "Inconsist length"

    for i in range(len(predRes)):
        if labels[i] in predRes[i]:
            res[0] += 1
        res[1] += 1

    return float(res[0])/res[1]

def countNodes(nodeIndex):
    #nodeIndex:[0,0,1,1,2,2]
    nodeSet = set(tuple(nodeIndex))
    indices = [0 for i in range(len(nodeSet))]
    for i in nodeSet:
        indices[i] = nodeIndex.count(i)
    return indices

def nodeCount2NodeIndex(indices):
    nodeIndex = []
    count = 0
    for i in indices:
        tmp = [j for j in range(count, count + i)]
        count += i
        nodeIndex.append(tmp)
    nodeIndex = np.array(list(itertools.zip_longest(*nodeIndex, fillvalue=count))).T
    return nodeIndex

def computeTopNBySeq(preds, labels, indices, n):
    # preds: [0,1,0,0,1,0,0,1], labels: [0,1,0,0,1,0,0,1], indices: [2,3,3]

    predRes = returnTopNRes(preds, indices, n)
    labels = returnTopNRes(labels, indices, 1)
    res = [0,0]

    for i in range(len(predRes)):
        if labels[i][0] in predRes[i]:
            res[0] += 1
        res[1] += 1

    return float(res[0])/res[1]

def dumpRes(info, filename):
    import pickle as pkl
    with open(filename,'wb') as f:
        pkl.dump(info, f)

def calCombineBySeq(locLabels, locPredRes, repLabels, repPredRes):
    TP, FP, TN, FN = 0, 1, 2, 3
    def retRes(label, pred):
        if label in pred:
            # 0 means no bug
            if label != 0:
                return TP
            else:
                return TN
        else:
            if label != 0:
                return FN
            else:
                return FP
    def retClassAcc(label, pred):
        if label != 0 and pred[0] != 0:
            return TP
        elif label == 0 and pred[0] == 0:
            return TN
        elif label == 0 and pred[0] != 0:
            return FP
        else:
            return FN

    res = [[0 for i in range(4)] for j in range(4)]
    classRes = [0 for i in range(4)]
    for i in range(len(locPredRes)):
        locRes = retRes(locLabels[i][0], locPredRes[i])
        repRes = retRes(repLabels[i][0], repPredRes[i])
        classLab = retClassAcc(locLabels[i][0], locPredRes[i])
        classRes[classLab] += 1
        res[locRes][repRes] += 1

    predCount = float(len(locPredRes))

    classAcc = (classRes[TP] + classRes[TN]) / sum(classRes)
    locAcc = (sum(res[TP]) + sum(res[TN])) / predCount
    # ideaily, res[TP][TN] should be 0
    #assert res[TP][TN] == 0
    repAcc = (res[TP][TP] + res[TP][TN]) / predCount
    combineAcc = (sum(res[TN]) + res[TP][TP] + res[TP][TN]) / predCount

    resArray = np.array(res)
    repSole = (sum(resArray[:,TP]) + sum(resArray[:,TN])) / predCount
    print("  class acc is: %.3f, loc acc is: %.3f, repair acc is: %.3f, combine acc is: %.3f" % (classAcc, locAcc, repAcc, combineAcc))
    #print("  solo repair acc is: %.3f" % (repSole))
    return res

def dumpPredInfo(locPreds, locLabels, repPreds, repLabels, indices):
    COUNT = 2
    indLook = indices[:COUNT*4]
    totalNum = sum(indLook)
    print(indLook)
    print(locPreds[:totalNum])
    print(locLabels[:totalNum])
    print(repPreds[:totalNum])
    print(repLabels[:totalNum])

def computeCombineBySeq(locPreds, locLabels, repPreds, repLabels, indices, n = 1, stride = 10):
    # locPreds: [0,1,0,0,1,0,0,1], locLabels: [0,1,0,0,1,0,0,1], indices: [2,3,3]
    # repPreds: [0,1,0,0,1,0,0,1], repLabels: [0,1,0,0,1,0,0,1], indices: [2,3,3]
    dumpPredInfo(locPreds, locLabels, repPreds, repLabels, indices)

    locPredRes = returnTopNRes(locPreds, indices, n)
    locLabels = returnTopNRes(locLabels, indices, 1)

    repPredRes = returnTopNRes(repPreds, indices, n)
    repLabels = returnTopNRes(repLabels, indices, 1)
    assert len(locPredRes) == len(repPredRes)

    res = calCombineBySeq(locLabels, locPredRes, repLabels, repPredRes)

    if stride == 0:
        return res

    sortedIndex = sorted(range(len(indices)), key=lambda k:indices[k])
    indicesList = []
    for i in range(int(100/stride)):
        begin = int(len(indices) * i * stride / 100)
        end = int(len(indices) * (i + 1) * stride / 100)
        indicesList.append(sortedIndex[begin:end])
    for indice in range(len(indicesList)):
        newLocPred = []
        newLocLabel = []
        newRepPred = []
        newRepLabel = []
        for i in indicesList[indice]:
            newLocPred.append(locPredRes[i])
            newLocLabel.append(locLabels[i])
            newRepPred.append(repPredRes[i])
            newRepLabel.append(repLabels[i])
        print(str(indice*stride) + " to " + str(indice*stride+stride) +": ", end="")
        res = calCombineBySeq(newLocLabel, newLocPred, newRepLabel, newRepPred)
    return res

def computeTopN(preds, indices, labels, n):
    # preds: [0,1,0,0,1,0,0,1], indices: [2,3,3], labels: [0,2,1]
    res = [0,0]
    pos = 0
    indLength = 0
    for i in indices:
        indLength += i
    assert indLength == len(preds), "Inconsist length"

    predRes = returnTopNRes(preds, indices, n)

    assert len(predRes) == len(labels), "Inconsist length"

    for i in range(len(predRes)):
        if labels[i] in predRes[i]:
            res[0] += 1
        res[1] += 1

    return float(res[0])/res[1]

def computeTopNWithoutInd(preds, labels, n):
    # preds: [[0,1],[0,0,1],[0,0,1], labels: [0,2,1]
    res = [0,0]
    pos = 0
    for i in range(len(preds)):
        predRes = preds[i]
        sortedIndex = sorted(range(len(predRes)), key=lambda k:predRes[k], reverse = True)
        predRes = sortedIndex[:n]
        if labels[i] in predRes:
            res[0] += 1
        res[1] += 1
    return float(res[0])/res[1]

def sortGraphSize(graphs, perc):
    sizesClean = []
    sizesBuggy = []
    for i in range(len(graphs)):
        graph = graphs[i]
        if graph["labels"][0]:
            sizesBuggy.append(len(graph["init"]))
            sizesClean.append(0)
        else:
            sizesClean.append(len(graph["init"]))
            sizesBuggy.append(0)
    return sizesBuggy, sizesClean

def sortGraphSizeInterval(graphs, perc):
    sizesClean = []
    sizesBuggy = []
    for i in range(len(graphs)):
        graph = graphs[i]
        size = 0
        ifBuggy = 0
        for k in graph:
            if k["insideinterval"] == 1:
                size += len(k["init"])
            else:
                ifBuggy = k["labels"][0]

        if ifBuggy:
            sizesBuggy.append(size)
            sizesClean.append(0)
        else:
            sizesClean.append(size)
            sizesBuggy.append(0)
    return sizesBuggy, sizesClean

def retProjIndex(data, projName):
    projNameIndex = []
    if isinstance(data[0], list):
        for i in range(len(data)):
            graph = data[i]
            for k in graph:
                if k["insideinterval"] == 0:
                    if k["projName"] == projName:
                        projNameIndex.append(i)
    else:
        for i in range(len(data)):
            graph = data[i]
            if graph["projName"] == projName:
                projNameIndex.append(i)
    return projNameIndex

def removeDataByIndex(graphs, projNameIndex):
    graphs[:] = [x for i, x in enumerate(graphs) if i not in projNameIndex]

def insertDataByIndex(source, target, projNameIndex, targetIsTraining=False):
    for i in projNameIndex:
        target.append(source[i])
    if targetIsTraining:
        np.random.shuffle(target)

def moveToTest(source, target, projName):
    projNameIndex = retProjIndex(source, projName)
    insertDataByIndex(source, target, projNameIndex)
    removeDataByIndex(source, projNameIndex)

def analyzeIntervalInfo(graphs):
    intervalSizeInfo = []
    numOfIntervalInfo = []
    for i in range(len(graphs)):
        graph = graphs[i]
        size = 0
        ifBuggy = 0
        numOfIntervals = 0
        intervalSize = []
        for k in graph:
            if k["insideinterval"] == 1:
                intervalSize.append(len(k["init"]))
                numOfIntervals += 1
        intervalSizeInfo += intervalSize
        numOfIntervalInfo.append(numOfIntervals)

    return intervalSizeInfo, numOfIntervalInfo

def analyzeGraphInfo(graphs):
    def stdDev(data, info):
        from scipy import stats
        n = len(data)
        mean = sum(data)/n
        variance = sum([((x - mean) ** 2) for x in data]) / n
        stddev = variance ** 0.5
        print(stats.describe(data))
        print("%s: mean: %.1f, stddev: %.1f, median: %d."%(info, mean, stddev, np.median(np.array(data))))
        data.sort()
        print(data)
    print("total num of graphs: %d" % (len(graphs)))
    intervalSizeInfo, numOfIntervalInfo = analyzeIntervalInfo(graphs)
    stdDev(intervalSizeInfo, "Size of intervals")
    stdDev(numOfIntervalInfo, "Number of intervals")


def filterGraphByPerc(graphs, perc):
    if perc == 100:
        return graphs
    if isinstance(graphs[0], list):
        sizesBuggy, sizesClean = sortGraphSizeInterval(graphs, perc)
    else:
        sizesBuggy, sizesClean = sortGraphSize(graphs, perc)
    #assert len(sizesBuggy) == len(sizesClean), "In Filter: inconsist length"

    sortedSizeIndexBuggy = sorted(range(len(sizesBuggy)), key=lambda k:sizesBuggy[k], reverse = True)
    splitIndex = max(1, int(float(perc)/len(sortedSizeIndexBuggy)))
    sortedSizeIndexBuggy = sortedSizeIndexBuggy[:splitIndex]

    sortedSizeIndexClean = sorted(range(len(sizesClean)), key=lambda k:sizesClean[k], reverse = True)
    splitIndex = max(1, int(float(perc)/len(sortedSizeIndexClean)))
    sortedSizeIndexClean = sortedSizeIndexClean[:splitIndex]
    sortedSizeIndex = sortedSizeIndexBuggy + sortedSizeIndexClean

    graphs[:] = [x for i, x in enumerate(graphs) if i in sortedSizeIndex]

def compute_rouge(predictions, targets):
    from rouge import Rouge
    predictions = [" ".join([str(x) for x in prediction]).lower() for prediction in predictions]
    predictions = [prediction if prediction else "EMPTY" for prediction in predictions]
    targets = [" ".join([str(x) for x in target]).lower() for target in targets]
    targets = [target if target else "EMPTY" for target in targets]
    rouge = Rouge()
    scores = rouge.get_scores(hyps=predictions, refs=targets, avg=True)
    return scores

def compute_seq_f1(references, translations, end_token=None, beta=1):
    """
    Computes BLEU for a evaluation set of translations
    Based on https://github.com/mast-group/convolutional-attention/blob/master/convolutional_attention/f1_evaluator.pyy
    """

    def find(vector, item):
        for i, element in enumerate(vector):
            if item == element:
                return i
        return -1

    total_f1 = 0
    total_prec = 0
    total_recall = 0
    newR = []
    newT = []
    for (reference, translation) in zip(references, translations):
        if end_token is not None:
            reference = reference[:find(reference, end_token)]
            translation = translation[:find(translation, end_token)]

        newR.append(reference)
        newT.append(translation)

        tp = 0
        ref = list(reference)
        for token in set(translation):
            if token in ref:
                ref.remove(token)
                tp += 1

        if len(translation) > 0:
            precision = tp / len(translation)
        else:
            precision = 0

        if len(reference) > 0:
            recall = tp / len(reference)
        else:
            recall = 0
        total_recall += recall
        total_prec += precision

        if precision + recall > 0:
            f1 = (1+beta**2) * precision * recall / ((beta**2 * precision) + recall)
        else:
            f1 = 0

        total_f1 += f1
    avgPrec = np.float32(total_prec / len(translations))
    avgRecall = np.float32(total_recall / len(translations))
    avgF1 = np.float32(total_f1 / len(translations))

    scores = compute_rouge(newR, newT)
    r1_f1 = scores['rouge-1']['f']
    r2_f1 = scores['rouge-2']['f']
    rl_f1 = scores['rouge-l']['f']

    return r1_f1, r2_f1, rl_f1
    #return avgPrec, avgRecall, avgF1

def compute_seq_f1_portion(pred, y, numOfNodes, end_token, stride = 10):
    if stride == 0:
        return
    sortedIndex = sorted(range(len(numOfNodes)), key=lambda k:numOfNodes[k])
    indicesList = []
    for i in range(int(100/stride)):
        begin = int(len(numOfNodes) * i * stride / 100)
        end = int(len(numOfNodes) * (i + 1) * stride / 100)
        indicesList.append(sortedIndex[begin:end])
    for indice in range(len(indicesList)):
        newPred = []
        newY = []
        for i in indicesList[indice]:
            newPred.append(pred[i])
            newY.append(y[i])

        precision, recall, avgF1 = compute_seq_f1(newPred, newY, end_token)

        f1 = 2 * precision * recall / (precision + recall)
        print(str(indice*stride) + " to " + str(indice*stride+stride) +": ",  end="")
        #print("  (precision is: %.5f, recall is: %.5f, f1 is: %.5f, avgF1 is %.5f)" % (precision, recall, avgF1, f1))
        print("  (r1-f is: %.5f, r2-f is: %.5f, fl-f is: %.5f)" % (precision, recall, avgF1))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
            ))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden


def convertInitialNodeRep(mask, x, embeddings, sl, initial_node_representation):
    revMask = tf.math.logical_not(mask)
    x = tf.nn.embedding_lookup(embeddings, x)
    trace_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
    dyoutputs, dyfinalstate = tf.nn.dynamic_rnn(trace_encoder_cell, x, sequence_length=sl, dtype=tf.float32)
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask,1)
    nodeRepRNN = tf.multiply(dyfinalstate[1], mask)
    revMask = tf.cast(revMask, tf.float32)
    revMask = tf.expand_dims(revMask,1)
    nodeInitRep = tf.multiply(initial_node_representation, revMask)
    return dyfinalstate[1]
    #self.placeholders['initial_node_representation'] = tf.math.add(nodeRepRNN, nodeInitRep)
    #self.placeholders['initial_node_representation'] = nodeInitRep

def loopRNN(mask, x, sl, embeddings, final_node_representations, name):
    with tf.variable_scope(name):
        x = tf.nn.embedding_lookup(embeddings, x)
        trace_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        #dyoutputs, dyfinalstate = tf.nn.dynamic_rnn(trace_encoder_cell, x, sequence_length=sl, dtype=tf.float32, initial_state=final_node_representations)
        dyoutputs, dyfinalstate = tf.nn.dynamic_rnn(trace_encoder_cell, x, sequence_length=sl, dtype=tf.float32)
        #self.placeholders['initial_node_representation'] = dyfinalstate[1]
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask,1)
        nodeRepRNN = tf.multiply(dyfinalstate[1], mask)
    return tf.math.add(nodeRepRNN, final_node_representations)


def stateRNN(final_node_representations, hidden_size, labelInd, numOfValidNodesInGraph, gatherNodeIndice, name):
    #self.placeholders['intraLabelIndex']: [[1,2,3,10], [4,5,10,10], [6,7,8,9]]
    #self.placeholders['numOfValidNodesInGraph']: [3,2,4]
    #converted_node_representation: [[1],[2]], assume the shape is [None, 1]
    finalRP = tf.pad(final_node_representations, [[0, 1], [0, 0]], "CONSTANT")
    finalRP = tf.nn.embedding_lookup(finalRP, labelInd) #[3,4,h_size]
    with tf.variable_scope(name):
        trace_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        dyoutputs, dyfinalstate = tf.nn.dynamic_rnn(trace_encoder_cell, finalRP, sequence_length=numOfValidNodesInGraph, dtype=tf.float32) # dyoutputs: [3,4,100], dyfinalstate: [3,100]
    iniNP = tf.gather_nd(dyoutputs, gatherNodeIndice)
    return iniNP, dyfinalstate[1]

def useRNN(hidden_size, labelInd, numOfValidNodesInGraph, x, embeddings, gatherNodeIndice):
    #converted_node_representation: [[1],[2]], assume the shape is [None, 1]
    x = tf.slice(x, [0, 0], [-1, 1]) # [None, 1]
    x = tf.concat([x,[[0]]], axis=0) #[None+1, 1]
    x = tf.nn.embedding_lookup(x, labelInd) # [3,4,1]
    x = tf.squeeze(x, axis=2) #[3,4]
    x = tf.nn.embedding_lookup(embeddings, x) #[3,4,h_size]
    trace_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    dyoutputs, dyfinalstate = tf.nn.dynamic_rnn(trace_encoder_cell, x, sequence_length=numOfValidNodesInGraph, dtype=tf.float32) # dyoutputs: [3,4,100]

    initial_node_representation = tf.gather_nd(dyoutputs, gatherNodeIndice)
    '''
    # these codes are used to implement the above two lines, but failed.
    dyoutputs = tf.reshape(dyoutputs, shape=[-1, hidden_size])
    seqIndex = tf.reshape(intraLabelIndex, shape=[-1])
    padNum = tf.math.reduce_max(seqIndex)
    mask = tf.not_equal(seqIndex, padNum)
    initial_node_representation = tf.boolean_mask(dyoutputs, mask)
    '''
    return initial_node_representation

def computeLocTmp(model, tokenRep, finalRep, ifslice = False):
    name = "colsplit"
    hidden_size = model.params['hidden_size']
    with tf.variable_scope(name):
        with tf.variable_scope("W1"):
            W1 = MLP(hidden_size, hidden_size, [], 1.0)
        with tf.variable_scope("W2"):
            W2 = MLP(hidden_size, hidden_size, [], 1.0)
        with tf.variable_scope("W3"):
            W3 = MLP(hidden_size, 1, [], 1.0)
        with tf.variable_scope("W4"):
            W4 = MLP(hidden_size, 1, [], 1.0)

        mask = model.placeholders['nodeMask']
        nodeIndexInGraph = model.placeholders['nodeIndexInGraph'] #[#Nodes,1]
        H2 = tf.gather_nd(finalRep, nodeIndexInGraph) #[#Nodes, 100]
        H2 = tf.boolean_mask(H2, mask)

        H1 = tf.boolean_mask(tokenRep, mask)

        E1 = W1(H1) + W2(H2) # [#Nodes, 100]
        loc = W3(E1) #[#Nodes, 1]
        repair = W4(E1) #[#Nodes, 1]
        if ifslice:
            E2 = tf.concat([loc, repair], axis = 1)
            newE2 = tf.transpose(E2) #[2, #Nodes]
            return (loc, repair, newE2)
        else:
            newE2 = tf.transpose(E2) #[2, #Nodes]
            return newE2

def computeLoc(model, tokenRep, finalRep, ifslice = False):
    name = "colsplit"
    hidden_size = model.params['hidden_size']
    with tf.variable_scope(name):
        with tf.variable_scope("W1"):
            W1 = MLP(hidden_size, hidden_size, [], 1.0)
        with tf.variable_scope("W2"):
            W2 = MLP(hidden_size, hidden_size, [], 1.0)
        with tf.variable_scope("W3"):
            W3 = MLP(hidden_size, 2, [], 1.0)

        mask = model.placeholders['nodeMask']
        nodeIndexInGraph = model.placeholders['nodeIndexInGraph'] #[#Nodes,1]
        H2 = tf.gather_nd(finalRep, nodeIndexInGraph) #[#Nodes, 100]
        H2 = tf.boolean_mask(H2, mask)

        H1 = tf.boolean_mask(tokenRep, mask)

        E1 = W1(H1) + W2(H2) # [#Nodes, 100]
        E2 = W3(E1) #[#Nodes, 2]
        if ifslice:
            loc = tf.slice(E2, [0, 0], [-1, 1]) # [None, 1]
            repair = tf.slice(E2, [0, 1], [-1, 1]) # [None, 1]
            newE2 = tf.transpose(E2) #[2, #Nodes]
            return (loc, repair, newE2)
        else:
            newE2 = tf.transpose(E2) #[2, #Nodes]
            return newE2

def pickNodesByGraphs(pred, label, nodeIndex):
    #pred: [#Nodes, 1], label: [#Nodes], nodeIndex: [#Graphs, MAXLength]
    pred = tf.concat([pred,[[-10000000000]]], axis=0) #[#Node+1, 1]
    pred = tf.nn.embedding_lookup(pred, nodeIndex) # [#Graphs,MAXLength,1]
    label = tf.concat([label,[0]], axis=0) #[#Node+1]
    label = tf.nn.embedding_lookup(label, nodeIndex) # [#Graphs,MAXLength]
    return pred, label

def computeLoss3(model, task_id, internal_id, compMisuse):
    # compMisuse: -1: both, 0: rep only, 1: misuse only
    useAttention = model.params.get("use_attention")
    if useAttention != None and useAttention:
        model.calAttention()
        model.computeAttentionWeight()
    finalRep = model.ops['final_node_representations']
    nodeIndex = model.placeholders["nodeIndex"]

    pred = computeLoc(model, finalRep, model.finalStat, True)
    loc = pred[0]
    misuse_pos_str = 'misuse_pos'
    repair_pos_str = 'repair_pos'
    if compMisuse == 0:
        misuse_pos_str = 'repair_pos'
    elif compMisuse == 1:
        repair_pos_str = 'misuse_pos'
    misuse_pos = tf.to_int64(model.placeholders[misuse_pos_str])
    loc, misuse_pick = pickNodesByGraphs(loc, misuse_pos, nodeIndex)
    loc = tf.squeeze(loc, axis=2)
    loc = tf.nn.softmax(loc)
    repair = pred[1]
    repair_pos = tf.to_int64(model.placeholders[repair_pos_str])
    repair, repair_pick = pickNodesByGraphs(repair, repair_pos, nodeIndex)
    repair = tf.squeeze(repair, axis=2)
    repair = tf.nn.softmax(repair)


    costLoc = tf.nn.softmax_cross_entropy_with_logits(logits=loc, labels=misuse_pick)
    costRep = tf.nn.softmax_cross_entropy_with_logits(logits=repair, labels=repair_pick)
    cost = costLoc + costRep
    #cost = costRep
    #cost = costLoc

    tmpPred = tf.argmax(loc,1)
    tmpPred = tf.equal(tmpPred, 0)
    tmpY = tf.argmax(misuse_pick,1)
    tmpY = tf.equal(tmpY, 0)
    correct_pred = tf.equal(tmpPred, tmpY)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    model.ops["accuracy_task%i" % task_id] = accuracy

    model.ops["losses"].append(cost)

    accInfo1 = tf.stack([misuse_pos, repair_pos])
    model.ops["acc_info"] = [pred[2], accInfo1, model.placeholders['numTokenInGraph']]


def doMask(arr, mask):
    assert len(arr) == len(mask)
    newArr = []
    for i in range(len(arr)):
        if mask[i] == True:
            newArr.append(arr[i])
    return newArr

if __name__ == "__main__":
    locPreds  = [1,0,0,0,0,1,0,0,0,0,1,0]
    locLabels = [1,0,0,0,1,0,0,0,0,0,1,0]
    repPreds  = [1,0,0,0,0,1,0,0,0,0,1,0]
    repLabels = [1,0,0,0,0,1,0,0,0,0,1,0]
    indices = [4,4,4]
    computeCombineBySeq(locPreds, locLabels, repPreds, repLabels, indices, n = 1)
    locPreds  = [1,0,0,0,0,1,0,0,0,0,1,0]
    locLabels = [0,1,0,0,1,0,0,0,0,0,1,0]
    repPreds  = [1,0,0,0,0,1,0,0,0,0,1,0]
    repLabels = [0,0,1,0,0,1,0,0,0,0,1,0]
    indices = [4,4,4]
    computeCombineBySeq(locPreds, locLabels, repPreds, repLabels, indices, n = 1)
    # locPreds: [0,1,0,0,1,0,0,1], locLabels: [0,1,0,0,1,0,0,1], indices: [2,3,3]
    # repPreds: [0,1,0,0,1,0,0,1], repLabels: [0,1,0,0,1,0,0,1], indices: [2,3,3]
