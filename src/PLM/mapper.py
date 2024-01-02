import numpy as np
import torch,sys,pickle

# Feature mappers convert graphs into matrices given lexicon and vsm
class FeatureMapper:
	def __init__(self, HOME, vsm, lexicon, mwa=False):
		self.HOME = HOME
		self.vsm = vsm
		self.lexicon = lexicon
		self.multiword_averaging = mwa
		
	def get_matrix(self,graph_list):
		ngraphs = len(graph_list)
		Y = []
		YFE = []
		T = []
		lemmapos = []
		gid = []
		j = 0
		print(len(graph_list),graph_list[-1].sid)
		'''
		DESCRIPTION OF TRAIN-TYPE GRAPH OBJECT'S ATTRIBUTES
		G 			    : digraph
		predicate_head  : integer (probably node ID)
		predicate_nodes : list of integers (probably node IDs)
		roles			: list (empty sometimes)
		sent 			: string
		gid				: integer (graph ID)
		sid				: integer (sentence ID) Note: each sentence creates more than 1 graph due to presence of multiple targets
  		'''
		for g in graph_list:
			targetWord, frameID, onehot_FEid = self.get_repr(g, self.lexicon)
			'''
			print(f"shape of the target word is -> {targetWord.shape}")
			print(f"this is the frameID, which we use as output label - {frameID}")
			print(f"this is the shape of the one-hot encodede vector for fining FEs which we use as the label for FE output -> {onehot_FEid.shape}")
			'''
			T += [targetWord]
			Y += [frameID]
			YFE += [onehot_FEid]
			lemmapos += [g.get_predicate_head()["lemmapos"]]
			gid += [g.gid]
			if (j % 1000 == 0):
				print(j,'/',ngraphs)
				sys.stdout.flush()
			#if (j > 0) and (j % 100 == 0):
			#	break
			j += 1
		T = torch.stack(T)	
		Y = np.array(Y, dtype=np.int)
		YFE = np.array(YFE, dtype=np.float32)
		return T, Y, YFE, lemmapos, gid
	
	def save_BERT(self, graph_list, HOME, corpus, embsl):
		print("inside save_bert")
		j = 0
		ngraphs = len(graph_list)
		wemb_dict = {}
		for g in graph_list:
			
			# we get :
			# sk = sentence key
			# s  = sentec
			# w = word embedding
			sk, s, we, se, ael, at = self.get_WrdsEmbd(g)
			if (j % 1000 == 0):
				print(j,'/',ngraphs)
				sys.stdout.flush()
			
			wemb_dict[sk] = (s, we)
			j += 1
		with open(HOME+'/data/corpora/'+corpus+'.sentences.'+embsl+'.pkl', 'wb') as handle:
			pickle.dump(wemb_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

class Mapper_One(FeatureMapper): # ONLY WHEN PRECOMPUTING BERT TENSORS
	def get_WrdsEmbd(self, graph):
		sent = graph.sent.split(" ")
		sent = [s.lower() for s in sent]
		#
		# structe of sent -> ['token1','token2',....'tokenN']
		#
		wembs, semb, ael, attns = self.vsm.compute(sent)
		sentkey = "_".join(sent)
		# semb + ael + attns are empty values
		return sentkey, sent, wembs,semb, ael, attns
	
def getAll_embedding(wordList,embeddings,sent,ndxs):
    if (ndxs[0] == -1):
        ndxs[0] = 0
    v0  = np.zeros(768,dtype=np.float32)
    res = [v0] * (max(ndxs)+1)
    assert len(wordList) == len(ndxs)
    for word,ndx in zip(wordList,ndxs):
        word = word.lower()
        #calls the 'get' function of myPLM class in extras.py
        res[ndx] = embeddings.get(word,ndx,sent)
    #print(f"The shape of the fetched word embedding is - {res[0].shape}")    
    return res
	
class Mapper_two(FeatureMapper):
    def get_repr(self,graph,lexicon):
        words = graph.sent.split(" ")
        ndxs = list(range(len(words)))
        x = getAll_embedding(words, self.vsm, words, ndxs)
        x = np.asarray(x)
        # shape of x -> (sent_length, hidden size)
        pad_requirement = x.shape[0]
        x = torch.from_numpy(np.pad(x,((0,512-pad_requirement),(0,0))))
        #print(f"Shape of x after padding is -> {x.shape}")
        
        # Dynamic window computation
        vrb = graph.find_parent_verb(graph.predicate_head)
        '''
        if (vrb != -1):
            deps = list(graph.get_direct_dependents(vrb))
            if (deps == []):
                WIN = [-10, 10]
            else:
                deps = np.array(deps)-1
                md = min(deps)
                Md = max(deps)
                gph = graph.predicate_head - 1
                if (gph < md):
                    md = gph
                if (gph > Md):
                    Md = gph
                WIN = [md - gph, Md - gph] 
        else:
            WIN = [-1000, 1000] # ALL SENTENCE!
        '''    
        WIN = [-1000, 1000] # ALL SENTENCE!
        # M
        '''
        -------------- the predicate head info
			The predicate head is a dict which contains the following :
			1) word 	3) frame
			2) pos  	4) lemmapos
		'''
        predicate_head  = graph.get_predicate_head()
        head_word       = [predicate_head['word'].lower(),]
        head_word_ndx   = [graph.predicate_head -1]
        target_word     = graph.get_predicate_node_words()
        target_word_ndx = [gpn-1 for gpn in graph.predicate_nodes]
        '''
        print(f"The head word is          -> {head_word}")
        print(f"The index of head word is -> {head_word_ndx}\n")
        print(f"The target word is        -> {target_word}")
        print(f"The target words index is -> {target_word_ndx}")
        '''
        if (self.multiword_averaging):
            val = np.array([min(target_word_ndx),max(target_word_ndx),len(ndxs)],dtype=np.int32)
            m   = torch.from_numpy(val)
        else:
            val = np.array([min(head_word_ndx),max(head_word_ndx),len(ndxs)],dtype=np.int32)
            m   = torch.from_numpy(val)
        # target is the sum of all features of the target word
        #print(f" this is what the slicer is -> {m[0]}:{m[1]+1}: ")
          
        target  = torch.sum(x[m[0]:m[1]+1,:], dim = 0)
        frameID = self.lexicon.get_id(predicate_head['frame'])
        fesE    = np.zeros((self.lexicon.get_number_of_FEs(),), dtype= np.float32)  # size (1148,) representing the Total of FEs
        
        atleaset1 = False
        for n in graph.G.nodes:
            role = graph.G.node[n].get("role","_")
            if (role != "_"):
                fesE[self.lexicon.get_FEid(role)]=1.0  # hot encode fesE at the ID number of the role FE
                atleaset1 = True
        if (not atleaset1):
            fesE[self.lexicon.get_FEid('NONE')] = 1.0  
        #lemmapos = graph.get_predicate_head()["lemmapos"]
        '''
			target  - contains the word embedding of the frame
			frameID - contains the ID of the frame
			fesE    - is a hot encoded vector for the frame element for the respective frame
        '''    
        return target,frameID,fesE  