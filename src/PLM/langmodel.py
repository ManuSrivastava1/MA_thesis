from torch.utils.data import TensorDataset,SequentialSampler,DataLoader
from transformers import AutoConfig,AutoTokenizer,AutoModel,DebertaV2Tokenizer,DebertaV2Config,DebertaV2Model
from transformers import T5Config,T5Tokenizer,T5EncoderModel
import sys,pickle,codecs,torch
import numpy as np

def load_from_HF(path,case):
    if ("electra" in path):
        c = AutoConfig.from_pretrained(path,output_hidden_states = True,output_attentions=True)
        t = AutoTokenizer.from_pretrained(path,do_lower_case = not case)
        m = AutoModel.from_pretrained(path,config=c)
    elif ("gte" in path):
        c = AutoConfig.from_pretrained(path,output_hidden_states = True,output_attentions=True)
        t = AutoTokenizer.from_pretrained(path,cls_token ="[CLS]",sep_token="[SEP]", do_lower_case = not case)
        m = AutoModel.from_pretrained(path,config=c)
    elif("deberta" in path):
        c = DebertaV2Config.from_pretrained(path,output_hidden_states = True,output_attentions=True)
        t = DebertaV2Tokenizer.from_pretrained(path,cls_token ="[CLS]",sep_token="[SEP]", do_lower_case = not case)
        m = DebertaV2Model(config=c)
    elif("t5" in path):
        c = T5Config.from_pretrained(path,output_hidden_states = True,output_attentions=True)
        t = T5Tokenizer.from_pretrained(path, do_lower_case = not case,model_max_length = 512)
        m = T5EncoderModel.from_pretrained(path,config=c)
    else:
        print("using degfault case setting for loading PLM and Tokeniser")
        c = AutoConfig.from_pretrained(path,output_hidden_states = True,output_attentions=True)
        t = AutoTokenizer.from_pretrained(path,do_lower_case = not case)
        m = AutoModel.from_pretrained(path,config=c)
    return c,t,m

def plm_setup(func):
    def wrapper(self,src,cased=False):
        # Set up parameters for the PLM being used
        layers_included     = '-1,-2,-3,-4'
        self.max_seq_length = 512
        self.batch_size     = 1
        self.layer_indexes  = layers_included = [int(x) for x in layers_included.split(",")]
        self.plm_path       = src
        self.case           = cased
        # Set up the CUDA device
        if torch.cuda.is_available():
            print(f"Device being used -> {torch.cuda.get_device_name()}")
            device_option = 0
            self.device         = torch.device(f"cuda:{device_option}")
        else:
            print(f"Device being used -> {torch.cuda.get_device_name}")
            self.device         = torch.device("cpu")

        func(self, src, self.case )  # Call the original method
    return wrapper

class PLM:
    @plm_setup
    def __init__(self,path,MultilingualCased=False):
        C,T,M          = load_from_HF(path,self.case)
        self.tokeniser = T
        self.model     = M.to(self.device)
        self.model.eval()

    def encodeTokens(self,input):
        # structe of input -> ['token1','token2',....'tokenN']
        tkn_lst,tkn_map = [],[]
        if ("electra" or "gte" or "t5" or "deberta" in self.plm_path):
            # -------------- adding special tokens and creating token position map
            tkn_lst.append("[CLS]")
            for wrd in input:
                tkn_map.append(len(tkn_lst))
                tkn_lst.extend(self.tokeniser.tokenize(wrd))
            if len(tkn_lst) > (self.max_seq_length-1):
                tkn_lst = tkn_lst[0:self.max_seq_length-1]
            tkn_lst.append("[SEP]")
        else:
            print("ERROR :Selected PLM is either unknown or its encoding method is not set")
            sys.exit()
        
        ids      = self.tokeniser.convert_tokens_to_ids(tkn_lst)
        attnMask = [1]*len(ids)
        # -------------- adding PADDING to everything
        while len(ids) < self.max_seq_length:
            ids.append(0)
            attnMask.append(0)
        while len(tkn_map) < self.max_seq_length:
            tkn_map.append(0)
        output = {'TOK':tkn_lst,'IDS':ids,'ATTN':attnMask,'MAP':tkn_map}

        return output
        
    def input2tensor(self,encoded_sent):
        ids      = torch.tensor([encoded_sent['IDS']], dtype=torch.long)
        attn     = torch.tensor([encoded_sent['ATTN']], dtype=torch.long)
        mapng    = torch.tensor([encoded_sent['MAP']], dtype=torch.long)
        sent_idx = torch.arange(ids.size(0), dtype =torch.long)
        
        '''
            Shape of ids,attn and mapng should be - [1,max_seq_length]
        '''
        evalData = TensorDataset(ids,attn,mapng,sent_idx)
        samplr   = SequentialSampler(evalData)
        dataLdr  = DataLoader(dataset=evalData,
                              sampler=samplr,
                              batch_size=self.batch_size)
        return dataLdr

    def getHiddenStates(self,dataloader,option=True):
        for inpIds,inpAttn,_,_ in dataloader:
            inpIds  = inpIds.to(self.device)
            inpAttn = inpAttn.to(self.device)
            
            modelOut = self.model(inpIds,attention_mask = inpAttn)
            '''
            print(type(modelOut))
            print(modelOut['last_hidden_state'].shape)
            print("Number of hidden states are - ",len(modelOut['hidden_states']))
            print("Dimension of hidden state is - ",modelOut['hidden_states'][1].shape)
            print("Number of attention layers is - ",len(modelOut['attentions']))
            print("Size of attention layer is - ",modelOut['attentions'][1].shape)
            '''
            if (option == True) :
                
                slctdLayers = modelOut['hidden_states'][1:]
                avrg        = torch.stack([slctdLayers[i] for i in self.layer_indexes])
                avrg_mean   = avrg.mean(0)/len(self.layer_indexes)
            elif (option == False):
                
                slctdLayers = modelOut['last_hidden_state']
                avrg_mean   = slctdLayers

        return avrg_mean          

    def WordEmbd(self,wrds_embds,dataloader,sentences):
        s = []
        
        for _,_,inpMap,inpIdx in dataloader:
            for i,idx in enumerate(inpIdx):
                for j in range(len(sentences[idx])):
                    if inpMap[i,j] < (self.max_seq_length-1):
                        s.append(wrds_embds[i,inpMap[i,j]].clone().detach().cpu())
                    else:
                        s.append(wrds_embds[i,inpMap[i,(self.max_seq_length-1)]].clone().detach().cpu())
        # shape of each element of s is (hidden size)
        # length of s list depend on the length of the input sentence
        staked_s = [s]
        return staked_s
    
    def make_wordEmbeddings(self,sentences):
        #print("This is how the sentence looks when sent to sentence embedding maker function\n")
        #print(sentences)
        # structe of sentences -> [['token1','token2',....'tokenN']]
        # notices its a list of lists
        for oneSent in sentences:
            encdSent       = self.encodeTokens(oneSent)
            inp_Dataloader = self.input2tensor(encdSent)
            hiddenStates   = self.getHiddenStates(inp_Dataloader)
            # shape of hiddenStates - [1,512,hiddenSize]
            word_embedings = self.WordEmbd(hiddenStates,inp_Dataloader,sentences)
        return word_embedings,[[]],[[]],[[]]
    
class myPLM:
    def __init__(self,src,mode):
        if mode == 'create':
            print('Creating CONTEXTUAL VECTOR model with',src)
            self.plm = self.plm = PLM(src)
        else:
            print('loading precomputed CONTEXTUAL EMBEDDINGS dictionary from',src)
            with open(src,'rb') as handle:
                self.plm = pickle.load(handle)
            self.vdim = self.plm[next(iter(self.plm))][1][0].shape[0]

    def compute(self,sent):
        '''
       		We break the sentence into tokens and make them all to lower-case
			Then they are passed to get sentence embeddings.
        '''
        sent = [s.lower() for s in sent]
        #print(sent)
        # structe of sent -> ['token1','token2',....'tokenN']
        w_embs, s_embs, ael, attns = self.plm.make_wordEmbeddings([sent])
        # w_embs is a list of list which contains only 1 list which is of the word embeddings 
        return w_embs[0],s_embs[0],ael[0],attns[0]

    def get(sefl,word,indx,sent):
        pass
            
'''
m = PLM('google/electra-base-discriminator')
string = ["This","is","a","test"]
m.make_wordEmbeddings([string])
print('=======')
#myM = myPLM(src='google/electra-base-discriminator',mode='create')
#myM.compute(string)
'''
