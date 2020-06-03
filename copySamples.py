import subprocess
import os
import ROOT

massggh = ['115', '120', '124', '125', '126', '130', '135', '140', '145', '150', '155', '160', '165', '170', '175', '180', '190', '200', '210', '230', '250', '270', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '900', '1000', '1500', '2000', '2500', '3000', '4000', '5000']

massvbf = ['115', '120', '124', '125', '126', '130', '135', '140', '145', '150', '155', '160', '165', '170', '175', '180', '190', '200', '210', '230', '250', '270', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '900', '1000', '1500', '2000', '2500', '3000', '4000', '5000']

# massggh = massvbf = ['115', '1000']
def getSampleFiles(inputDir,Sample,absPath=False,rooFilePrefix='latino_',FromPostProc=False):

    #### SETUP DISK ACCESS ####
    if not '/eos/' in  inputDir and '/store/' in inputDir:
        Dir = '/eos/cms/' + inputDir
    else:                          
        Dir = inputDir
    if '/eos/cms/' in inputDir:
        absPath=True
        xrootdPath='root://eoscms.cern.ch/'
            
    lsCmd='ls ' 
        


    ##### Now get the files for Sample
    fileCmd = lsCmd+Dir+'/'+rooFilePrefix+Sample+'.root'
    if 'root://' in inputDir:
        fileCmd = lsCmd+Dir+'/ | grep '+rooFilePrefix+Sample+'.root' 
    proc    = subprocess.Popen(fileCmd, stderr = subprocess.PIPE,stdout = subprocess.PIPE, shell = True)
    out,err = proc.communicate()
    Files   = out.split()
    if len(Files) == 0 :
        fileCmd = lsCmd+Dir+'/'+rooFilePrefix+Sample+'__part*.root'
        if 'root://' in inputDir:
            fileCmd = lsCmd+Dir+'/ | grep '+rooFilePrefix+Sample+'__part | grep root'
        proc    = subprocess.Popen(fileCmd, stderr = subprocess.PIPE,stdout = subprocess.PIPE, shell = True)
        out,err = proc.communicate()
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




treeBaseDir = '/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano'


def makeMCDirectory(var=''):
    if var:
        return os.path.join(treeBaseDir, mcProduction, mcSteps.format(var='__' + var))
    else:
        return os.path.join(treeBaseDir, mcProduction, mcSteps.format(var=''))

mcProduction = 'Fall2017_102X_nAODv5_Full2017v6'
mcSteps = 'MCl1loose2017v6__MCCorr2017v6__Semilep2017'

mcDirectory = makeMCDirectory()



files=[]
# files = nanoGetSampleFiles(mcDirectory, 'WJetsToLNu-0J')[0:1] +\
#         nanoGetSampleFiles(mcDirectory, 'WJetsToLNu-1J')[0:5] +\
#         nanoGetSampleFiles(mcDirectory, 'WJetsToLNu-2J')[0:20]

for mass in massggh:
    paths = nanoGetSampleFiles(mcDirectory, 'GluGluHToWWToLNuQQ_M'+mass)
    if isinstance(paths, str):
        files.append(paths)
        print("hello, this is a bug")
    else: files.append(paths[0])

for mass in massvbf:
    paths = nanoGetSampleFiles(mcDirectory, 'VBFHToWWToLNuQQ_M'+mass)
    if isinstance(paths, str):
        files.append(paths)
        print("hello, this is a bug")
    else: files.append(paths[0])



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
