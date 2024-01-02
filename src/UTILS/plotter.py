import matplotlib.pyplot as plt
import numpy as np

class metricLogger(object):
    def __init__(self):
        self.epochCount               = 0
        self.plotPath                 = None
        # ============ training loss values
        self.combined_loss            = []
        self.frameClassification_loss = []
        self.frameElement_clsf_loss   = []
        self.graphEmbedding_loss      = []
        # ============ Accuracy values
        self.trainingSet_acc          = []
        self.validationSet_acc        = []
        self.testSet_acc              = []
        # ============ AUC and AP values
        self.area_uc                  = []
        self.average_p                = []

    def log_losses(self,l,l1,l2,l3):
        self.combined_loss.append(l)
        self.frameClassification_loss.append(l1)
        self.frameElement_clsf_loss.append(l2)
        self.graphEmbedding_loss.append(l3)

    def log_accuracy(self,tr_ac,vl_ac,ts_ac):
        self.trainingSet_acc.append(tr_ac)
        self.validationSet_acc.append(vl_ac)
        self.testSet_acc.append(ts_ac)

    def log_auc_ap(self,auc,ap): 
        self.area_uc.append(auc)
        self.average_p.append(ap)   

    def store_final_epoch(self,num):
        self.epochCount = num    
    
    def set_plot_filePath(self,path):
        self.plotPath = path

    def plot(self):
        # Plotting Loss
        epochs = np.arange(1, self.epochCount + 1)
        plt.figure(figsize=(15, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.combined_loss, label='Combined Loss', color='blue')
        plt.plot(epochs, self.frameClassification_loss, label='F - Classification Loss', color='green')
        plt.plot(epochs, self.frameElement_clsf_loss, label='FE - Classification Loss', color='orange')
        plt.plot(epochs, self.graphEmbedding_loss, label='AGE Loss', color='red')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.trainingSet_acc, label='Training Accuracy', color='purple')
        plt.plot(epochs, self.validationSet_acc, label='Validation Accuracy', color='red')
        plt.plot(epochs, self.testSet_acc, label='Test Accuracy', color='orange')
        plt.title('Accuracy Log')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Save or Show
        if self.plotPath:
            plt.savefig(self.plotPath + "_loss_accuracy.png")
            print(f"Loss and Accuracy Plots saved at: {self.plotPath}_loss_accuracy.png")
        else:
            plt.show()

        # Plotting Metrics
        plt.figure(figsize=(10, 5))

        # Average Precision and AUC Plot
        plt.plot(epochs, self.average_p, label='Average Precision', color='cyan')
        plt.plot(epochs, self.area_uc, label='Area Under Curve', color='magenta')
        plt.title('Training Metrics of AGE')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics Value')
        plt.legend()
        plt.grid(True)

        # Save or Show
        if self.plotPath:
            plt.savefig(self.plotPath + "_metrics.png")
            print(f"Metrics Plots saved at: {self.plotPath}_metrics.png")
        else:
            plt.show()