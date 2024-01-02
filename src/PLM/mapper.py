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