import torch,pickle,math,sys
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from AGE import *
from UTILS import metricLogger

def countParams(model):
	tp  = sum(p.numel() for p in model.parameters() if p.requires_grad)
	ntp = sum(p.numel() for p in model.parameters() if not p.requires_grad)
	return tp, ntp

class myDataset(torch.utils.data.Dataset):
	def __init__(self, T, y, yFE, ncl):
		'Initialization'
		self.T   = T
		self.y   = y
		self.yFE = yFE
		self.ncl = ncl

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.T)

	def __getitem__(self, index):
		'Generates one sample of data'
		T   = self.T[index]
		y   = self.y[index]
		yFE = self.yFE[index]
		return T, y, yFE

class baseClf:
    def __init__(self, lexicon,all_unknown=False, num_components=False, max_sampled=False, num_epochs=False):
        self.clf = None
        self.lexicon = lexicon
        self.all_unknown = all_unknown
        self.num_components = num_components
        self.max_sampled = max_sampled
        self.num_epochs = num_epochs
        
class Net(torch.nn.Module):
    def __init__(self,inpShape,frmCount,embSize,emb2fr,FEcount,device):
        super(Net,self).__init__()
        self.inpShape   = inpShape   # input size of NN
        self.outF       = frmCount   # output size of NN
        self.outFE      = frmCount   # output of NN
        self.hidden     = 256
        self.gnn        = LinTrans(layers=1,dims=[768,embSize])
        self.device     = device
        
        self.endX       = torch.tensor(emb2fr).to(device)
        self.prj_T      = torch.nn.Linear(self.inpShape[-1],self.hidden)
        self.prj_L      = torch.nn.Linear(embSize,self.hidden)
        self.fc3        = torch.nn.Linear(self.hidden,FEcount)
        
        self.outW       = Parameter(torch.Tensor(self.outF,self.hidden))
        self.outb       = Parameter(torch.Tensor(self.outFE))
        
        # Performing Weight initialisation
        torch.nn.init.kaiming_uniform_(self.outW, a=math.sqrt(5))
        fan_in, _       = torch.nn.init._calculate_fan_in_and_fan_out(self.outW)
        bound           = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.outb, -bound, bound)
        
    def forward(self,target,AGE_inx):
        AGE_e           = self.gnn(AGE_inx)
        
        h               = torch.tanh(self.prj_T(target))
        E               = torch.tanh(self.prj_L(AGE_e[self.endX]))
        
        hp              = h.unsqueeze(1).repeat(1,E.shape[0],1)
        Ep              = E.unsqueeze(0).repeat(h.shape[0],1,1)
        joint_prj       = torch.mul(hp, Ep)
        output          = torch.mul(joint_prj, self.outW.unsqueeze(0).repeat(h.shape[0],1,1))
        output_frame    = torch.sum(output, dim=-1)
        output_frame    = torch.add(output_frame, self.outb.unsqueeze(0).repeat(h.shape[0],1))
        output_frameE   = self.fc3(h)
        
        return output_frame,output_frameE

class Classifier(baseClf):
    def __init__(self, lexicon, all_unknown=False, num_components=False, max_sampled=False, num_epochs=False,
                 inF=None, outF=None, model_file=None, lexdir=None, dataset=None):
        super().__init__(lexicon, all_unknown, num_components, max_sampled, num_epochs)
        
        torch.set_printoptions(profile="full")
        self.logger     = metricLogger()
        #save argument values
        self.model_file = model_file
        self.lexdir     = lexdir
        self.dataset    = dataset
        self.fetr_Shape = inF
        self.numFrms    = outF
        self.gamma1     = 0.5
        self.gamma2     = 0.1
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.outFE      = lexicon.get_number_of_FEs() 
        self.outE       = 768
        
        self.fr2id      = lexicon.frameToId
        self.id2fr      = lexicon.idToFrame
        # AGE dict being loaded
        afile           = self.lexdir+'/AGE_data/'+self.dataset+'U.dict'
        embsNdx         = pickle.load(open(afile,"rb"))
        self.emb2fr     = [0]*len(embsNdx)
        for k,v in embsNdx.items():
            self.emb2fr[self.fr2id[k]] = v
        
        self.model      = Net(self.fetr_Shape,
                         self.numFrms,
                         self.outE,
                         self.emb2fr,
                         self.outFE,
                         self.device)
        self.model.to(self.device)
        
        self.loss_c     = torch.nn.CrossEntropyLoss()   # lossFunction for Frame classification
        self.loss_f     = torch.nn.BCEWithLogitsLoss()  # lossFunction for FrameElement classification
        self.lr         = 2e-05
        self.wgt_decay  = 1e-04
        self.optimiser  = torch.optim.AdamW(self.model.parameters(), lr = self.lr, weight_decay = self.wgt_decay)

        tp, ntp = countParams(self.model)
        print('This is the LexDir  -> ',self.lexdir,'\n')
        print('This is the Dataset -> ',self.dataset,'\n')
        print('This is the type of inF -> ',type(self.fetr_Shape))
        print("===========exploring the AGE dict")
        print(f"length of emb2fr based on AGE dict is -> {len(self.emb2fr)}")
        print('\n',self.model)
        print(f"Number of trainable parameters -> {tp}\nNumber of NON trainable parameters -> {ntp}")
    
    def LR_scheduler(self,current_epoch):
            print("modifying LR and WEIGHT DECAY")
            self.lr        *= 0.8
            self.wgt_decay *= 0.8 
            self.optimiser  = torch.optim.AdamW(self.model.parameters(), lr = self.lr, weight_decay = self.wgt_decay) 
            
    def train(self,t_tr,y_tr,yFE_tr,lp_tr,
              t_dv,y_dv,yFE_dv,lp_dv,
              t_ts,y_ts,yFE_ts,lp_ts):
        # Hyper-parameters for AGE
        args_gnnlayers = 8
        args_linlayers = 1
        args_epochs    = 500
        args_lr        = 0.001
        args_upth_st   = 0.0011
        args_lowth_st  = 0.1
        args_upth_ed   = 0.001
        args_lowth_ed  = 0.5
        args_upd       = 1
        AGE_bs         = 256
        batch_size     = 16
        
        # Making the path for loading the data for AGE
        using          = self.dataset+'U'
        dataset        = self.lexdir+'/AGE_data'
        print(f"GNN Module using -> {using}\nusing the path -> {dataset}")
        
        # Loading the data using the AGE loader
        ename = ""
        adj, features, _, idx_train, idx_val, idx_test = load_data(dataset,using,ename)
        n_nodes, feat_dim                              = features.shape
        
        # STARTING THE ADJACCENCY MATRIX MANIPULATION
        # Store original adjacency matrix (without diagonal entries) for calculating ROC score later
        adj      = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        adj_orig = adj
        
        # MASKING EDGES FOR GETTING THE TRAINING MATRIX
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        adj           = adj_train
        numRows_inADJ = adj.shape[0]  # STORE THE NUMBER OF ROWS IN ADJ -> # 1022  * 1022
        
        # NORMALISING THE ADJ ACCORDING TO AGE_paper
        adj_norm_s    = preprocess_graph(adj, args_gnnlayers, norm='sym', renorm=True)
        sm_fea_s      = sp.csr_matrix(features).toarray()
        
        print('AGE: Laplacian Smoothing...')
        for a in adj_norm_s:
            sm_fea_s  = a.dot(sm_fea_s)
        
        adj_1st            = (adj + sp.eye(numRows_inADJ)).toarray()
        adj_label          = torch.FloatTensor(adj_1st)
        sm_fea_s           = torch.FloatTensor(sm_fea_s)
        adj_label          = adj_label.reshape([-1,])
        inx                = sm_fea_s.clone().detach().requires_grad_(True)#.cuda()
        adj_label          = adj_label#.cuda()
        pos_num            = len(adj.indices)
        neg_num            = n_nodes*n_nodes-pos_num
        up_eta             = (args_upth_ed - args_upth_st) / (args_epochs/args_upd)
        low_eta            = (args_lowth_ed - args_lowth_st) / (args_epochs/args_upd)
        pos_inds, neg_inds = update_similarity(normalize(sm_fea_s.numpy()), args_upth_st, args_lowth_st, pos_num, neg_num)
        upth, lowth        = update_threshold(args_upth_st, args_lowth_st, up_eta, low_eta)
        pos_inds_cuda      = torch.LongTensor(pos_inds)#.cuda()

        # ==========================================================================================
        
        # making the dataset using our custom  class and the loader
        trainset    = myDataset(t_tr, y_tr, yFE_tr, self.numFrms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        # SETUP THE TRAINING LOOP
        # initiating the variables for keeping track of training
        epoch       = 1
        bestVacc    = 0
        pat         = 0
        avgV        = [0.0]*10
        train_loss  = 1.0
        
        # setting stopage condition
        #while ((pat < 10) or (train_loss_c > 0.2)):
        while epoch < 3:
            
            train_loss = train_loss_c = train_loss_e = train_loss_f = 0.0
            self.model.train()
            
            for t, targetF, targetFE in trainloader:
                
                self.optimiser.zero_grad()
                # we select some negatively related and positively related nodeIDs and make our sample with them
                sampled_neg  = torch.LongTensor(np.random.choice(neg_inds, size=AGE_bs))#.cuda()
                sampled_pos  = torch.LongTensor(np.random.choice(pos_inds_cuda.cpu(), size=AGE_bs))#.cuda()
                sampled_inds = torch.cat((sampled_pos, sampled_neg), 0)
                
                x_ind        = sampled_inds // n_nodes
                y_ind        = sampled_inds % n_nodes
                x            = torch.index_select(inx, 0, x_ind)
                y            = torch.index_select(inx, 0, y_ind)
                z_x          = self.model.gnn(x)                 # this give problem if we change the dims of gnn
                z_y          = self.model.gnn(y)
                
                batch_label  = torch.cat((torch.ones(AGE_bs), torch.zeros(AGE_bs)))#.cuda()
                batch_pred   = self.model.gnn.dcs(z_x, z_y)
                
                loss_e       = torch.nn.functional.binary_cross_entropy_with_logits(batch_pred, batch_label)

                t, targetF, targetFE = t.to(self.device), targetF.to(self.device), targetFE.to(self.device)
                # forward pass
                #print(f"shape of t -> {t.shape}{type(t)}\nshape of inx -> {inx.shape}{type(inx)}")
                output_c, output_f   = self.model(t, inx.to(self.device))
                
                # calculate the losses
                loss_c      = self.loss_c(output_c, targetF.to(torch.int64))
                loss_f      = self.loss_f(output_f, targetFE)
                loss        = self.gamma2 * (self.gamma1 * loss_c + (1-self.gamma1) * loss_e) + (1-self.gamma2) * loss_f
                #print("Cummalative loss -> ",loss)
                #print("Loss.item()      -> ",loss.item())
                # backward pass
                loss.backward()
                self.optimiser.step()
                # update running training loss
                train_loss   += loss.item()*t.size(0)
                train_loss_c += loss_c.item()*t.size(0)
                train_loss_e += loss_e.item()*t.size(0)
                train_loss_f += loss_f.item()*t.size(0)
                
            
            sys.stdout.flush()    
            self.model.eval()
            E                  = self.model.gnn(inx)
            hidden_emb         = E.cpu().data.numpy()
            upth, lowth        = update_threshold(upth, lowth, up_eta, low_eta)
            pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num)
            pos_inds_cuda      = torch.LongTensor(pos_inds)#.cuda()
            
            # calculate average loss over an epoch
            train_loss   = train_loss/len(trainloader.sampler)
            train_loss_c = train_loss_c/len(trainloader.sampler)
            train_loss_e = train_loss_e/len(trainloader.sampler)  # is the loss of the gnn(AGE) i think
            train_loss_f = train_loss_f/len(trainloader.sampler)
            self.logger.log_losses(train_loss,train_loss_c,train_loss_f,train_loss_e)
            # EVAL GNN
            val_auc, val_ap = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
            self.logger.log_auc_ap(val_auc,val_ap)
            # EVAL FRM_CLF on dev and test splits
            TRacc = self.evaluate(t_tr, y_tr, lp_tr, inx)
            Vacc  = self.evaluate(t_dv, y_dv, lp_dv, inx)
            Tacc  = self.evaluate(t_ts, y_ts, lp_ts, inx)
            self.logger.log_accuracy(TRacc,Vacc,Tacc)
            avgV.pop(0)
            avgV.append(Vacc)
            
            # print training statistics
            print("AGE: {}, train_loss_gae={:.5f}, val_auc={:.5f}, val_ap={:.5f}, VAL_={:.5f}".format(
                    epoch, loss_e, val_auc, val_ap, val_auc+val_ap))
            print('Epoch: %3d  Loss: %6.5f (%6.5f,%6.5f,%6.5f)  Accs: train-> %8.7f, dev-> %8.7f(%6.5f), Test_Accs-> %8.7f  Pat: %d' % (epoch, train_loss, train_loss_c, train_loss_e, train_loss_f, TRacc, Vacc, sum(avgV)/len(avgV), Tacc, pat), end='')
            
            '''
            if epoch % 150 == 0:
                self.LR_scheduler(epoch)
            '''
            
            Vacc = sum(avgV) / len(avgV)  # set validation_acc to averarage
            if (Vacc > bestVacc):
                torch.save({'model':self.model.state_dict(), 'inx': inx, 'lexicon': self.lexicon}, self.model_file)
                print('*')
                bestVacc = Vacc
                pat = 0
            else:
                if (epoch > 10):
                    pat += 1
                print()
            self.logger.store_final_epoch(epoch)
            epoch += 1
        
        self.logger.set_plot_filePath("../")
        self.logger.plot()

    def predict(self,T, lemmapos, inx):
        available_frames = self.lexicon.get_available_frame_ids(lemmapos)  # get available frames from lexicon
        ambig            = self.lexicon.is_ambiguous(lemmapos)
        unknown          = self.lexicon.is_unknown(lemmapos)
        
        bestScore = None
        bestClass = None
        if unknown or self.all_unknown:  # the all_unknown setting renders all lemma.pos unknown!
            available_frames = self.lexicon.get_all_frame_ids()  # if the lemma.pos is unknown, search in all frames
        else:
            if not ambig:
                # if the LU is known and has only one frame, just return it. Even if there is no data for this LU (!)
                bestClass = available_frames[0]
        
        # JOIN POSSIBLE FEs FOR EACH AVAILABLE FRAME
        available_FEs = []
        for af in available_frames:
            available_FEs += self.lexicon.frameToFE[af]
        available_FEs.append(self.lexicon.FEToId['NONE'])
        available_FEs = list(set(available_FEs))
        
        # DOING INFERENCE
        self.model.eval()
        T        = T.unsqueeze(0)
        o_c, o_f = self.model(T.to(self.device), inx.to(self.device))
        y        = torch.squeeze(o_c)
        o_f      = torch.squeeze(o_f)
        
        if (bestClass == None):
            expA_F     = []
            inv_expA_F = {}
            for cl in available_frames:
                if (cl < len(y)):
                    score = y[cl]
                    if ((bestScore is None) or (score >= bestScore)):
                        bestScore = score
                        bestClass = cl
                        
        return bestClass, (o_f > 0.0).int(), available_FEs
    
    def evaluate(self, T, y, lp, inx):
        total   = 0
        correct = 0
        for t_, y_true, lp_ in zip(T, y, lp):
            y_predicted, _, _ = self.predict(t_, lp_, inx)
            correct          += int(y_true == y_predicted)
            total            += 1
        acc     = correct / total if total != 0 else 0
        return acc