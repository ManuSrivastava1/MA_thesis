import torch,sys,argparse,os,time
from sklearn.metrics import jaccard_score

from UTILS import ResourceManager,ReportManager,Score,Lexicon,SetUp,get_graphs
from PLM import myPLM,Mapper_two
from model import Classifier
'''
******************** DOCUMENTATION ********************
    This module contains three functions namely,
    1) parse2
    2) setup
    3) doScoring
    
    This module is called from the FrameID.sh script file and it passes 
    four values to it - 'mode','home_dir','plm name' and 'FN version'. These values 
    are taken in using the 'function_1' which shows this info to the user for 
    verification.
    Then it passes these values to 'function_2' where we perform the following tasks,
    FIRST  -> Load the lexicon of the respective FN version being used
    SECOND -> set the lexicon 
    THIRD  -> create the PLM object
    FOURTH -> create the mapper for creating the sentence embeddings for our data
    FIFTH  -> split the raw data into 3 splits and make graphs from it.
    SIXTH  -> pass the graph list to the PLM object to get sentence embeddings for each graph
              and save this new data as a pickle file.
******************** ************** ********************
'''
def parse2():
    parser = argparse.ArgumentParser(description="Process arguments vectorising data using PLMs")

    # Add argument for setting the mode
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="set the mode for the programme -- train/test",
    )

    # Add argument for the home directory path
    parser.add_argument(
        "--home",
        type=str,
        required=True,
        help="set the path for home directory",
    )

    # Add argument for name of the PLM used for embedding
    parser.add_argument(
        "--embedding_name",
        type=str,
        required=True,
        help="set the name of the embedding used in vectorsing process",
    )

    # Add argument for name of the framenet Dataset version used during the experiment
    parser.add_argument(
        "--dataset_v",
        type=str,
        required=True,
        help="set the name of the framenet Dataset version used during the experiment",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values
    mode = args.mode
    embeddings_name = args.embedding_name
    dataset = args.dataset_v
    home_dir   = args.home

    # Print the arguments just to verify 
    print(f"Mode                 : {mode}")
    print(f"Home Path            : {home_dir}")
    print(f"Embedding's type     : {embeddings_name}")
    print(f"FN version of Dataset: {dataset}")

    # make resource manager and report manager
    print("Starting resource manager")
    sources = ResourceManager(home_dir)
    print("Initializing reporters")
    reports = ReportManager(sources.out)
    
    return sources,reports,mode,home_dir,embeddings_name,dataset

def loadGraphs(rsrcM,rprtM):
    corpus_train,corpus_dev,corpus_test = 'train','dev','test'
    g_train = get_graphs(*rsrcM.get_corpus(corpus_train))
    g_test = get_graphs(*rsrcM.get_corpus(corpus_test))
    g_dev = get_graphs(*rsrcM.get_corpus(corpus_dev))
    
    rprtM.conll_reporter_train.report(g_train)
    return g_train,g_test,g_dev
    
def makeSetups(mode,home,emb_name,dataset_v):
    # we set some values for our setup
    lexicons = ['joined_lex']     # lexicon to use (mind the all_unknown setting!)
    multiword_averaging = [True]  # treatment of multiword predicates, false - use head embedding, true - use avg
    all_unknown = [False]         # makes the lexicon treat all LU as unknown, corresponds to the no-lex setting
    repeats = 1                   # set the number of times to repeat the experiment to get the results    
    
    # we load the required word embeddings for the data
    if ('electra' in emb_name):
        if ('electra' in emb_name):
            embSfx = f'.sentences.{emb_name}.pkl'
            pretrained_lm = ['google/electra-large-discriminator']
        elif ('electra_b' in emb_name):
            embSfx = f'.sentences.{emb_name}.pkl'
            pretrained_lm = ['google/electra-base-discriminator']
    elif ('gte' in emb_name):
        embSfx = f'.sentences.{emb_name}.pkl'
        pretrained_lm = ['thenlper/gte-large']
    elif ('deberta' in emb_name):
        if ('debertaV2' in emb_name):
            embSfx = f'.sentences.{emb_name}.pkl'
            pretrained_lm = ['microsoft/deberta-v2-xlarge']
        elif ('debertav3' in emb_name):
            embSfx = f'.sentences.{emb_name}.pkl'
            pretrained_lm = ['microsoft/deberta-v3-large']
        else:
            embSfx = f'.sentences.{emb_name}.pkl'
            pretrained_lm = ['microsoft/deberta-v3-base']
    elif ('t5' in emb_name):
        if ('3b' in emb_name):
            embSfx = f'.sentences.{emb_name}.pkl'
            pretrained_lm = ['google/t5-3b']
        elif ('xl' in emb_name):
            embSfx = f'.sentences.{emb_name}.pkl'
            pretrained_lm = ['google/t5-v1_1-xl']
    else:
        embSfx = f'.sentences.bert.pkl'
        pretrained_lm = ['bert-base-uncased']
    print('The path to pickled file -> ',embSfx)
    print('The PLM being used       -> ',pretrained_lm)

    # we declare a variable for storing multiple setups if needed
    all_setups = []
    
    # we make setup for training and add it to the 'all_setup' 
    for lexicon in lexicons:
        for plm in pretrained_lm:
            for mwa in multiword_averaging:
                for all_unk in all_unknown:
                    for j in range(repeats):
                        all_setups += [SetUp(Classifier,Mapper_two,lexicon,plm,mwa,all_unk,None, None, None)]
    print(f"This is the setup of the experiment that will be run -> \n\n{all_setups[0]}")  
    return all_setups,embSfx
   
def loadLexicon(rsrcM,rprtM,setUp,myModel_path,mode,dataset_v):
    # we make the file paths for loading the lexicon
        #LEXICON_FULL = dataset_v+"_lexicon"
        #LEXICON_ONLY_TRAIN = "fnTrain_lexicon"
    LEXICON_ALLFN = dataset_v + "_AllFrames"
    
    lexicon = Lexicon()
    if mode == 'train':
        lexicon.load_from_list(rsrcM.get_lexicon(setUp.get_lexicon()),
                               rsrcM.get_lexicon(LEXICON_ALLFN))
    else:
        checkpoint = torch.load(myModel_path)
        lexicon = checkpoint['lexicon']
    
    rprtM.lexicon_reporter.report(lexicon)
    print(lexicon.frameToId)
    print(lexicon.get_number_of_frames())
    print(lexicon.FEToId)
    print(lexicon.get_number_of_FEs())
    
    return lexicon
 
def make_dataSplits(g_train,g_test,g_dev,homedir,lexicon,setup,embSfx):
    #WIP
    corpus_train,corpus_dev,corpus_test = 'train','dev','test'
    inpTrain,inpDev,inpTest = {},{},{}
    for split_type in [corpus_train,corpus_dev,corpus_test]:
        print(f'####{split_type}####')
        # we are loading the respective embeddings for each split
        vsm = myPLM(homedir+'/data/corpora/'+split_type+embSfx,'get')
        # we are passing the embeddings to Mapper_two
        mapper = setup.get_feat_extractor()(homedir, vsm, lexicon, mwa=setup.get_multiword_averaging())
        if ('train' in split_type):
            T_train, y_train, yFE_train, lemmapos_train, gid_train = mapper.get_matrix(g_train)
            inpTrain.update([('T-train',T_train), ('y-train',y_train), ('yFE-train',yFE_train), ('lemmaPos-train',lemmapos_train)])
        elif ('dev' in split_type):
            T_dev, y_dev, yFE_dev, lemmapos_dev, gid_dev = mapper.get_matrix(g_dev)
            inpDev.update([('T-dev',T_dev), ('y-dev',y_dev), ('yFE-dev',yFE_dev), ('lemmaPos-dev',lemmapos_dev)])
        elif ('test' in split_type):
            T_test, y_test, yFE_test, lemmapos_test, gid_test = mapper.get_matrix(g_test)
            inpTest.update([('T-test',T_test), ('y-test',y_test), ('yFE-test',yFE_test), ('lemmaPos-test',lemmapos_test),('gid-test',gid_test)])
            
    return inpTrain,inpDev,inpTest

def start_classifier(mode,rsrcM,lexicon,setUp,myModel_path,dataset_v,inpTrain,inpDev,inpTest):
    LEXICON_ALLFN = dataset_v+"_AllFrames"
    path_lexicon = os.path.split(rsrcM.get_lexicon(LEXICON_ALLFN))[0]
    print(f"this is the joined lexicon path -> {path_lexicon}")
    
    clf = setUp.get_clf()(lexicon,setUp.get_all_unknown(),setUp.get_num_components(),
                          setUp.get_max_sampled(),setUp.get_num_epochs(),
                          inpTrain['T-train'][0].shape,
                          lexicon.get_number_of_frames(),myModel_path,
                          path_lexicon,dataset_v)
    
    if (mode == 'train'):
        clf.train(inpTrain['T-train'],inpTrain['y-train'],inpTrain['yFE-train'],inpTrain['lemmaPos-train'],
                  inpDev['T-dev'],inpDev['y-dev'],inpDev['yFE-dev'],inpDev['lemmaPos-dev'],
                  inpTest['T-test'],inpTest['y-test'],inpTest['yFE-test'],inpTest['lemmaPos-test'])
    return clf

def do_evaluation(rprtM,setUp,g_test,modelPath):
    corpus_train = 'train'
    corpus_dev   = 'dev'
    corpus_test  = 'test'

    start_time = time.time()
    rprtM.set_config(setUp,corpus_train, corpus_test)
    rprtM.conll_reporter_test.report(g_test)
    print("Completed model evaluation \n")
    
    score = Score()         # storage for scores
    score_v = Score()       # storage for verb-only scores
    score_known = Score()   # storage for known lemma-only scores
    
    # Now we predict and compare
    print('Loading Trained model from -> ',modelPath)
    checkpoint = torch.load(modelPath)
    clf.model.load_state_dict(checkpoint['model'])
    inx = checkpoint['inx']
    js_num = js_den = 0
    for t, y_true, fe_true, lemmapos, gid, g in zip(inpTest['T-test'], inpTest['y-test'], inpTest['yFE-test'], inpTest['lemmaPos-test'], inpTest['gid-test'], g_test):
        y_predicted, fe_predicted, fe_indexes = clf.predict(t, lemmapos, inx)
        fe_predicted = fe_predicted.detach().cpu().numpy()
        fe_predicted = fe_predicted[fe_indexes]
        fe_true = fe_true[fe_indexes].astype(int)
        fe_pl = [idx for idx, val in enumerate(fe_predicted) if val != 0]
        fe_tl = [idx for idx, val in enumerate(fe_true) if val != 0]
        js_num += jaccard_score(fe_true, fe_predicted)
        js_den += 1
        y_true = y_true.item()
        correct = y_true == y_predicted

        score.consume(correct, lexicon.is_ambiguous(lemmapos), lexicon.is_unknown(lemmapos), y_true)
        if lemmapos.endswith(".v"):
            score_v.consume(correct, lexicon.is_ambiguous(lemmapos), lexicon.is_unknown(lemmapos), y_true)
        if not lexicon.is_unknown(lemmapos):
            score_known.consume(correct, lexicon.is_ambiguous(lemmapos), lexicon.is_unknown(lemmapos), y_true)

        rprtM.result_reporter.report(gid, g, lemmapos, y_predicted, y_true, lexicon)
        
    return score,js_num,js_den,start_time
    

if __name__ == "__main__":
    print('PyTorch Version:',torch.__version__,'\n')
    rsrcM,rprtM,mode,home,embeddings_name,dataset_v = parse2()
    g_train,g_test,g_dev = loadGraphs(rsrcM,rprtM)
    all_Setups,embSfx    = makeSetups(mode,home,embeddings_name,dataset_v)
    
    # Now we set the filePaths for the models which will be saved from the experiments
    current_setup = 0
    for setUp in all_Setups:
        current_setup           += 1
        start_time              = time.time()
        myModel_path            = f'{rsrcM.root}/model{current_setup}.bin'
        lexicon                 = loadLexicon(rsrcM,rprtM,setUp,myModel_path,mode,dataset_v)
        inpTrain,inpDev,inpTest = make_dataSplits(g_train,g_test,g_dev,home,lexicon,setUp,embSfx)
        
        print("Model file's path ->:",myModel_path)
        print(f"Shape of T-train is -> {inpTrain['T-train'].shape}\nShape of One example in T-train is -> {inpTrain['T-train'][0].shape}")
        
        clf                          = start_classifier(mode,rsrcM,lexicon,setUp,myModel_path,dataset_v,inpTrain,inpDev,inpTest)
        
        score,jsNum,jsDen,start_time = do_evaluation(rprtM,setUp,g_test,myModel_path)
        fe_js                        = jsNum/jsDen
        rprtM.summary_reporter.report("train", "test", setUp, score, fe_js, time.time() - start_time)
        
        print ("============ STATUS: - setup", current_setup, "/", len(all_Setups))