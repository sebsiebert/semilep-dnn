import subprocess
import os
import ROOT
import argparse


treeBaseDir = '/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano'


def getSampleFiles(inputDir,Sample,absPath=False,rooFilePrefix='latino_',FromPostProc=False):

    #### SETUP DISK ACCESS ####
    if not '/eos/' in  inputDir and '/store/' in inputDir:
        Dir = '/eos/cms/' + inputDir
    else:                          
        Dir = inputDir
    if '/eos/cms/' in inputDir:
        absPath=True
        # xrootdPath='root://eoscms.cern.ch/'
            
    lsCmd='ls ' 
        


    ##### Now get the files for Sample
    fileCmd = lsCmd+Dir+'/'+rooFilePrefix+Sample+'.root'
    if 'root://' in inputDir:
        fileCmd = lsCmd+Dir+'/ | grep '+rooFilePrefix+Sample+'.root' 
    proc    = subprocess.Popen(fileCmd, stderr = subprocess.PIPE,stdout = subprocess.PIPE, shell = True)
    out,_ = proc.communicate()
    Files   = out.split()
    if len(Files) == 0 :
        fileCmd = lsCmd+Dir+'/'+rooFilePrefix+Sample+'__part*.root'
        if 'root://' in inputDir:
            fileCmd = lsCmd+Dir+'/ | grep '+rooFilePrefix+Sample+'__part | grep root'
        proc    = subprocess.Popen(fileCmd, stderr = subprocess.PIPE,stdout = subprocess.PIPE, shell = True)
        out,_ = proc.communicate()
        Files   = out.split()
    if len(Files) == 0 and not FromPostProc :
        print ('ERROR: No files found for sample ',Sample,' in directory ',Dir)
        exit() 
    FileTarget = []
    for iFile in Files:
        if absPath : FileTarget.append(iFile)
        else       : FileTarget.append(os.path.basename(iFile)) 
    return FileTarget



def nanoGetSampleFiles(inputDir, sample):
        return getSampleFiles(inputDir, sample, True, 'nanoLatino_')



def makeMCDirectory(var=''):
    if var:
        return os.path.join(treeBaseDir, mcProduction, mcSteps.format(var='__' + var))
    else:
        return os.path.join(treeBaseDir, mcProduction, mcSteps.format(var=''))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Copy branches needed for DNN to personal eos.")
    parser.add_argument('samples',
                        choices=["wjets", "top", "ggf", "vbf", "all"])

    samples = parser.parse_args().samples


    massggh = ['115', '120', '124', '125', '126', '130', '135', '140', '145', '150', '155', '160', '165', '170', '175', '180', '190', '200', '210', '230', '250', '270', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '900', '1000', '1500', '2000', '2500', '3000', '4000', '5000']

    massvbf = ['115', '120', '124', '125', '126', '130', '135', '140', '145', '150', '155', '160', '165', '170', '175', '180', '190', '200', '210', '230', '250', '270', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '900', '1000', '1500', '2000', '2500', '3000', '4000', '5000']



    mcProduction = 'Fall2017_102X_nAODv5_Full2017v6'
    mcSteps = 'MCl1loose2017v6__MCCorr2017v6__Semilep2017'

    mcDirectory = makeMCDirectory()



    files=[]
    if samples == "wjets" or samples == "all":
        files += nanoGetSampleFiles(mcDirectory, 'WJetsToLNu-1J')+\
                 nanoGetSampleFiles(mcDirectory, 'WJetsToLNu-2J')

    if samples == "ggf" or samples == "all":
        for mass in massggh:
            paths = nanoGetSampleFiles(mcDirectory, 'GluGluHToWWToLNuQQ_M'+mass)
            if isinstance(paths, str):
                files.append(paths)
                print("hello, this is a bug")
            else: files += paths

    if samples == "vbf" or samples == "all":
        for mass in massvbf:
            paths = nanoGetSampleFiles(mcDirectory, 'VBFHToWWToLNuQQ_M'+mass)
            if isinstance(paths, str):
                files.append(paths)
                print("hello, this is a bug")
            else: files += paths



    with open('branchList') as f:
        neededBranches = f.readlines()
    # remove whitespace
    neededBranches = [x.strip() for x in neededBranches]

    ROOT.ROOT.EnableImplicitMT()
    for filePath in files:
        print(filePath)
        oldFile = ROOT.TFile.Open(filePath)
        tree = oldFile.Get('Events')

        # deactivate all branches
        tree.SetBranchStatus("*", 0)

        # activate needed ones
        for branchName in neededBranches:
            tree.SetBranchStatus(branchName, 1)
        
        # save as new file
        newFile = ROOT.TFile(("/eos/user/s/ssiebert/rootfiles/" +
                                os.path.basename(filePath)), "recreate")
        newTree = tree.CloneTree()

        newFile.Write()

        oldFile.Close()
        newFile.Close()
