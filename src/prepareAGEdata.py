import pickle,sys,os,argparse
import numpy as np
from scipy.sparse import csr_matrix
from resources import ResourceManager

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
    embName     = " "
    if ('electra' in eName):
        if('base' in eName):
            embName   = '_electra_b'
        elif('large' in eName):
            embName   = '_electra_l'
    
    elif ('deberta' in eName):
        embName   = '_deberta'

    elif ('gte' in eName):
        embName   = '_gte_l'

    elif ('t5' in eName):
        if('3b' in eName):
            embName   = '_t5_3b'
        elif('v1_1' in eName):
            embName = '_t5_v1_1'
    else:
        print("unknown model used for making embedding, please set up its nomenclature")
        sys.exit()

    return root,dataVersion,embName

def makeFrameGraph(path,version,ename):
    #------------------ function documentation ------------------
    '''
    To be done !!!
    '''
    #------------------ function documentation ------------------
    '''
    k -> frame name
    v -> ndx or nodeID
    '''
    d   = {}   
    ndx = 0
    '''
    k -> ndx or nodeID of roote frame
    v -> ndx or nodeID of related frame
    '''
    e   = {}
    with open(path, "r") as file:
        print(type(file))
        for relation in file:
            related_nodes = relation.split()
            if (related_nodes[0] not in d):
                d[related_nodes[0]] = ndx
                ndx                += 1

            if (related_nodes[1] not in d):
                d[related_nodes[1]] = ndx
                ndx                += 1
            # structure:  < node_ndx  |  node_idx >
            #print(f'{related_nodes} --- {str(d[related_nodes[0]])} {str(d[related_nodes[1]])}')
            #print('\n')

            if (d[related_nodes[0]] in e.keys()):
                e[d[related_nodes[0]]] = e[d[related_nodes[0]]] + [d[related_nodes[1]]]
            else:
                e[d[related_nodes[0]]] = [d[related_nodes[1]]]
                
    print(e,len(e))
    print(len(d))
    #pickle.dump(d, open(sys.argv[1]+'.dict',"wb"))
    print('----------------------------------------')
    e[len(d)] = []
    e[len(d)+1] = []
    print(e,len(e))
    path = os.path.join(sources.resources,f'AGE_data/ind.{ename}_fn{version}U.graph')
    pickle.dump(e, open(path,"wb"))
    return d

def load_DefinitionEmbd(path,e):
    #------------------ function documentation ------------------
    '''
    ARGS     | D-TYPE
    path     | str
    e        | str

    -> path : is the path to the resources directory which contains
              the different sentence embeddings made using the selected
              PLM
    -> e : is the modifier attached to the generic string of the pickle file
           it gets used to select the right embeddings from the path.
    '''
    #------------------ function documentation ------------------
    slctdpath = os.path.join(path,f'fn1.5.taxonomyFramesEmbs{e}.pkl')
    print(f'\nSelected embedding path - {slctdpath}\n')
    with open(slctdpath, 'rb') as file:
        loaded_object = pickle.load(file)

    return loaded_object

def restructureData(embs,d):
    # MAKING 'allx'
    allx = np.zeros((len(d),len(embs['Killing'])))
    for k,v in embs.items():
        allx[d[k]] = np.array(v)
    allx = csr_matrix(allx)
    # MAKING 'x'
    x = allx[0:2,]
    x = csr_matrix(x)
    # MAKING 'y'
    y = np.zeros((2,2))
    y[0,0]    = y[1,1] = 1.0
    # MAKING 'ally'
    ally      = np.zeros((len(d),2))
    ally[0,0] = ally[1,1] = 1.0

    print(x.toarray().shape,allx.toarray().shape,y.shape)
    # MAKING 'tx'
    tx      = np.zeros((2,len(embs['Killing'])))
    tx[1,:] = 1.0
    tx      = csr_matrix(tx)
    # MAKING 'ty'
    ty = np.zeros((2,2))
    ty[0,0] = ty[1,1] = 1.0

    print(tx.toarray(),ty.shape)

    return x,y,allx,ally,tx,ty

def saveResources(x,y,allx,ally,tx,ty,sources,version,e,d):
    #------------------ function documentation ------------------
    '''
    ARGS     | D-TYPE
    x        | ndarray
    y        | ndarray
    allx     | csr_matrix
    ally     | ndarray
    tx       | ndarray
    ty       | ndarray
    sources  | <ResourceManager Object>
    version  | str
    e        | str
    d        | dict

    -> version : is the string which denotes which version of the frameNet
                 was used during the getting the frames and their definitons.
    -> e : is the string which is name modifier for the final pickle
           files. It is also a denoter of the plm used for making the
           frame defintion embeddings
    -> d : is the dictionary which contains the frames and its ID.
    '''
    #------------------ function documentation ------------------
    
    storageDirectory = os.path.join(sources.resources,'AGE_data')
    for var,data in zip(['x','y','allx','ally','tx','ty'],[x,y,allx,ally,tx,ty]):
        fullPath = os.path.join(storageDirectory,f'ind.{e}_fn{version}U.{var}')
        print(fullPath)
        pickle.dump(data,open(fullPath,"wb"))

    with open(os.path.join(storageDirectory,f"ind.fn{version}U.test.index"),"w") as f:
	    f.write(str(len(d))+'\n'+str(len(d)+1)+'\n')

if __name__ == "__main__":
    root,version,eName  = parse()
    sources             = ResourceManager(root)

    frameRelations_Path = os.path.join(sources.resources,f'AGE_data/fn{version}U')
    embeddings_Path     = os.path.join(sources.resources)

    d                   = makeFrameGraph(frameRelations_Path,version,eName)
    embs                = load_DefinitionEmbd(embeddings_Path,eName)
    x,y,allx,ally,tx,ty = restructureData(embs,d)
    
    saveResources(x,y,allx,ally,tx,ty,sources,version,eName,d)
    
    print("Completed AGE resource creation")