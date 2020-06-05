import ROOT
import tensorflow as tf
import inspect
import os
import array
from collections import OrderedDict
import random
import numpy as np

# list of files
# list of branch names
# lep1 *4
# lep2 *4
# jet1 *3
# jet2 *3
# jet3 *3
# jet4 *3
# fatjet1 *4
# fatjet2 *4
# largest_nonW_mjj *1
# PuppiMET_pt *1
# PuppiMET_phi *1
# some way to read them


def dataGenerator(batch_size, mode="train"):
    if mode == "train":
        offset = 5000
        step = 2
        maxPosition = 1e9
    elif mode == "validate":
        offset = 0
        step = 1
        maxPosition = 5000
    elif mode == "test":
        offset = 5001
        step = 2
        maxPosition = 1e9


    # FIXME: maybe add jet mass
    branchDict = OrderedDict([
    ('Lepton_pt',(2, 'f')), ('Lepton_eta',(2, 'f')),
    ('Lepton_phi',(2, 'f')), ('Lepton_pdgId',(2, 'i')),
    ('CleanJet_pt',(4, 'f')), ('CleanJet_eta',(4, 'f')),
    ('CleanJet_phi',(4, 'f')), ('CleanFatJet_pt',(2, 'f')),
    ('CleanFatJet_eta',(2, 'f')), ('CleanFatJet_phi',(2, 'f')),
    ('CleanFatJet_tau21',(2, 'f')), ('largest_nonW_mjj',(1, 'f')),
    ('PuppiMET_pt',(1, 'f')), ('PuppiMET_phi',(1, 'f'))
    ])
    
    branches = OrderedDict()
    for key, value in branchDict.items():
        num, type = value
        branches[key] = array.array(type, num*[0])
            

    basePath = "/home/sebastian/rootFiles/"
    fileList = os.listdir(basePath)
    ggfSet = {x for x in fileList if "GluGluH" in x}
    vbfSet = {x for x in fileList if "VBFH" in x}
    bgSet = {x for x in fileList if x not in ggfSet and x not in vbfSet}

    usedFiles = set()

    filesInUse = list()
    filesInUse += random.sample(ggfSet, 3)
    filesInUse += random.sample(vbfSet, 3)
    filesInUse += random.sample(bgSet, 4)
    usedFiles.update(filesInUse)

    tfiles = [ROOT.TFile(basePath+x) for x in filesInUse]
    trees = [x.Get('Events') for x in tfiles]
    treePositions = [offset]*len(tfiles)
    nEntries = [x.GetEntries() for x in trees]



    while True:

        features = []
        labels = []

        while len(features) < batch_size:
            # Get entry
            i = random.randrange(len(trees))

            # check if I need to replace with new file
            if (treePositions[i] >= nEntries[i] 
            or treePositions[i] >= maxPosition):
                if i < 2:
                    if ggfSet == set():
                        ggfSet = {x for x in usedFiles if "GluGluH" in x}
                        usedFiles = usedFiles - ggfSet
                    filesInUse[i] = random.choice(tuple(ggfSet - usedFiles))
                elif i < 4:
                    if vbfSet == set():
                        vbfSet = {x for x in usedFiles if "VBFH" in x}
                        usedFiles = usedFiles - vbfSet
                    filesInUse[i] = random.choice(tuple(vbfSet - usedFiles))
                else:
                    if bgSet == set():
                        bgSet = {x for x in usedFiles
                                if not "GluGluH" in x and not "VBFH" in x}
                        usedFiles = usedFiles - bgSet
                    filesInUse[i] = random.choice(tuple(bgSet  - usedFiles))

                tfiles[i].Close()
                tfiles[i] = ROOT.TFile(basePath+filesInUse[i])
                trees[i] = tfiles[i].Get('Events')
                treePositions[i] = offset
                nEntries[i] = trees[i].GetEntries()
                # print(filesInUse[i], nEntries[i])
            
            # get entry from selected file
            for key, value in branches.items():
                trees[i].SetBranchAddress(key, value)

            file = filesInUse[i]

            for _ in range(8):
                if (treePositions[i] >= nEntries[i] 
                or treePositions[i] >= maxPosition
                or len(features) >= batch_size):
                    break

                trees[i].GetEntry(treePositions[i])
                treePositions[i] += step

                # Append to batch lists
                tmp = []
                # print(branches["CleanJet_pt"])
                for value in branches.values():
                    tmp += list(value)

                tmp = np.array(tmp)
                features.append(tmp)

                tmpLabels = [ int("WJets"   in file),
                            int("GluGluH" in file)+int("VBFH" in file)]
                tmpLabels = np.array(tmpLabels)
                labels.append(tmpLabels)
        
        # features = np.array(features).reshape((batch_size, 31))
        # labels = np.array(labels).reshape((batch_size, 3))
        yield ( np.array(features),
                np.array(labels))



if __name__ == "__main__":
    i = 0
    for features, labels in dataGenerator(64):
        print(type(features), type(labels))
        print(features.shape, labels.shape)
        i+=1
        if i > 1: break
    
    print(next(dataGenerator(1)))

