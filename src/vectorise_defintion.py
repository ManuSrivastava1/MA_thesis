import torch,argparse,sys,pickle,os
from PLM import PLM
from UTILS import ResourceManager

def parse():
    parser = argparse.ArgumentParser(description="Process arguments vectorising data using PLMs")

    # Add argument for setting FrameNet version
    parser.add_argument(
        "--fn_v",
        type=str,
        required=True,
        help="frameNet version being for making the frame definition embeddings",
    )
    # Add argument for setting embedding maker
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="the name of the model used for making the sentence embeddings",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values
    dataVersion = args.fn_v
    eName       = args.model
    root        = f"../experiments/xp_{dataVersion}_08/frameid"

    return dataVersion,eName,root

def getModel(name):
    if ('electra' in name):
        if ('base' in name):
            embName   = '_electra_b'
            modelPath = 'google/electra-base-discriminator'
    
        elif ('large' in name):
            embName   = '_electra_l'
            modelPath = 'google/electra-large-discriminator' 
            return embName,modelPath
    elif 'deberta' in name:
        embName   = '_deberta'
        modelPath = 'microsoft/deberta-v3-large'
        return embName,modelPath
    elif 'gte' in name:
        embName   = '_gte_l'
        modelPath = 'thenlper/gte-large'
        return embName,modelPath
    elif 't5' in name:
        if '3b' in name:
            embName   = '_t5_3b'
            modelPath = 't5-3b'
            return embName,modelPath
        elif 'v1_1' in name:
            embName = '_t5_v1_1'
            modelPath = 'google/t5-v1_1-xl'
            return embName,modelPath
    else:
        print("unknown model used for making embedding, please set up its nomenclature")
        sys.exit()

def makeDefinitionsDict(path):
    '''
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in files:
        print(file)
    '''
    
    with open(path, 'r') as file:
        # Read the entire contents of the file
        content = file.read()
    
    lines = content.splitlines()
    Fdefs = {}

    for line in lines:
        condition = False
        for idx,w in enumerate(line):
            if not condition:
                '''
                NOTICE : the lines start with frame name without any spaces and a TAB and then the frame name + definition.
                '''
                if w == '\t':
                    condition = True
                    frame = line[:idx]
                    defin = line[idx:].strip()
                    Fdefs.update({frame:defin})

    return Fdefs

def getDefinitionEmbeddings(defintions,plm):
    embs = {}
    # iterate through definitions and make embeddings
    for Fname,Fdef in defintions.items():
        key   = Fname
        value = plm.make_defintionEmbeddings(Fdef)
        embs.update({key:value})
    return embs

if __name__ == "__main__":
    datasetV,Ename,root = parse()
    sources             = ResourceManager(root)
    print(Ename)
    print(type(Ename))
    emb,Mpath           = getModel(Ename)
    modelPath           = Mpath
    plm                 = PLM(modelPath)

    defintions          = makeDefinitionsDict(sources.get_frameDefintions(datasetV))
    embeddings          = getDefinitionEmbeddings(defintions,plm)

    
    name          = f"fn1.5.taxonomyFramesEmbs{emb}.pkl"
    print(name)

    with open(os.path.join(sources.resources,name),"wb") as fileName:
        pickle.dump(embeddings,fileName,protocol=pickle.HIGHEST_PROTOCOL)