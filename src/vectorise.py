import argparse
import sys

from UTILS import ResourceManager,get_graphs
from PLM import Mapper_One,myPLM
'''
******************** DOCUMENTATION ********************
    This module is used for the preprocessing of the 
    the dataset to make the vectorised pickle objects.
*******************************************************
'''
def parse():
    parser = argparse.ArgumentParser(description="Process arguments vectorising data using PLMs")

    # Add argument for setting FrameNet version
    parser.add_argument(
        "--fn_v",
        type=str,
        required=True,
        help="frameNet version being used in the experiments",
    )

    # Add argument for PLM model name
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the pre-trained language model (PLM)",
    )

    # Add argument for full path to PLM
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Full path to the pre-trained language model (PLM)",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values
    dataVersion = args.fn_v 
    model_name = args.model_name
    model_path = args.model_path
    home_dir   = f"../experiments/xp_{dataVersion}/frameid"

    # Your code to use model_name and model_path goes here
    print(f"Model Name : {model_name}")
    print(f"Model Path : {model_path}")
    print(f"Home Path  : {home_dir}")
    print(f"FN version : {dataVersion}")

    vectorise(home_dir,model_name,model_path)

def vectorise(home,PLMname,PLMpath):
    # variable names for creating data splits
    corpus_train = 'train'
    corpus_dev   = 'dev'
    corpus_test  = 'test'

    print(f"        Building vectors using -> {PLMname}")
    
    # The 'nameModifier' will be used for setting the name of the pickled files later and also will be used for calling it
    # But, we are using the PLMname directly for now
    #nameModifier = PLMname

	# STEP1 & 2 - create a resource manager for handling the file pathes,etc.
    print("        Starting resource manager")
    sources = ResourceManager(home)
    lexicon = None
    
	# STEP3 - create a class object for the PLM being used
	# 		  this contains 1. 'compute' : for getting sentece embeddings
	#						2. 'get' : for getting word embeddings
    
    plm = myPLM(PLMpath,'create')

    # STEP4 - create a class object which has a parent object called featuremapper
	#  		  and it contains our MODEL class object as well as lexicon
    
    mapper = Mapper_One(home, plm, lexicon)

    # STEP5 & 6 - we vectorize our dev dataset	
    for split_type in [corpus_dev,corpus_train,corpus_test]:
        # STEP5 - first we convert our sentences into dependecy graphs using the defined functions in 'data.py'
	    # 		  the data is fetches by the resource manager using the keyword stored in the corpus_dev variable.
        print('Vectorizing',split_type)
        sources_lst = sources.get_corpus(split_type)
        for item in sources_lst:
            print(item) 
        g_list = get_graphs(*sources.get_corpus(split_type))
        
	    # STEP6 - we using the mapper object created at step3 to iterate over the graphs and creates word embeddings for 
	    #			each word in the graph using the class object created at step 2. we use its compute method.
        mapper.save_BERT(g_list, home, split_type, PLMname)

if __name__ == "__main__":
    parse()
    print("Finished vecotrisation of dataset")


