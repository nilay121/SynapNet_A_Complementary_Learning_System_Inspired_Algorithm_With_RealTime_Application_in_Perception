from plasticModel import PlasticModel
from workingModel import WorkingModel 
from stableModel import StableModel
import numpy as np
import matplotlib.pyplot as plt
from cls_inhibition_algo import CustomInhibitStrategy
import torch
import torchvision
import pickle
import pandas as pd
import optuna
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
VOCSegmentation, Cityscapes, SBDataset, USPS, HMDB51, UCF101, \
CelebA, CORe50Dataset, TinyImagenet, CUB200, OpenLORIS
from torchvision import transforms
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from avalanche.benchmarks.utils import AvalancheDataset

from vae_model import VAE
from vae_training import Vae_Cls_Generator
from utils import utility_funcs,CustomDatasetForDataLoader
utility_funcs = utility_funcs()
class Hyperparametr:
    def __init__(self):
        pass

    def normalizer(self, data):
        zeroOneRange = ((data-data.min())/(data.max()-data.min()))
        minusOneOneRange = (zeroOneRange - 0.5)/0.5
        return minusOneOneRange
    
    def dataGeneration(self):
        
        tactileData = pd.read_csv("dataset/Known_data/BP_SensorData.csv", sep=",")
        labels = ['solid_square', 'solid_cylinder', 'stapler', 'airpods_case', 'gluestick', 'bottle', 'rectangular_block', 'hollow_cylinder',
       'airpods', 'mobile_phone', 'smart_watch', 'air_guage', 'big_cuboid', 'plier', 'ball']       
    
        modifiedData = []
        modifiedLabels = []
        for i in range(len(labels)):
            dataToTransform = np.array(tactileData.loc[tactileData['object'] == labels[i], ['SensorVal1', 'SensorVal2', 'SensorVal3', 'SensorVal4']])
            for j in range(0,dataToTransform.shape[0],150):
                tempTrans = dataToTransform[j:j+150].reshape(1,-1)
                modifiedLabels.append(i)
                modifiedData.append(tempTrans)
            print("All transformations DONE!!!")

        modifiedLabels = np.array(modifiedLabels)
        modifiedData = np.array(modifiedData).squeeze(1)

        train_indices, val_indices, _, _ = train_test_split(range(len(modifiedLabels)),modifiedLabels,stratify=modifiedLabels,
                test_size=0.20)
        trainData = modifiedData[train_indices]
        trainLabels = modifiedLabels[train_indices]

        testData = modifiedData[val_indices]
        testLabels = modifiedLabels[val_indices]

        X_normalizedTrain = self.normalizer(trainData)
        X_normalizedTest = self.normalizer(testData)
        X_normalizedTrain = X_normalizedTrain.astype("float32")
        X_normalizedTest = X_normalizedTest.astype("float32")

        trainData = CustomDatasetForDataLoader(data=X_normalizedTrain,targets=trainLabels)
        testData = CustomDatasetForDataLoader(data=X_normalizedTest,targets=testLabels)
        scenarioTrainVal = nc_benchmark(trainData, testData, n_experiences=5, shuffle=False, seed=9, task_labels=False)

        return scenarioTrainVal

    def dataPrepToPlot(self,acc_dict,exp_numb):
        y_stable=[]
        y_plastic=[]
        cls_output = []
        for i in range(0,len(acc_dict)):
            y_stable.append(np.array(list(acc_dict.values())[i][0].cpu()))
            y_plastic.append(np.array(list(acc_dict.values())[i][1].cpu()))
        '''
        The accuracy of the plastic model for the recent experiences are better than the stable model,
        whereas the accuracy of the stable model on the old experiences are better.
        We use both the stable model and the plastic model and store the accuracy that is the highest from 
        either of the model
        '''
        for outputs in range(len(y_stable)):
            '''
                taking the last output from the plastic model instead of the stable model
            '''
            if (outputs==(exp_numb-1)):
                cls_output.append(y_plastic[outputs])
            else:
                cls_output.append(y_stable[outputs])

        y_stable = np.array(y_stable)
        y_plastic = np.array(y_plastic)
        cls_output = np.array(cls_output)
        return np.round(y_stable,decimals=6),np.round(y_plastic,decimals=6),np.round(cls_output,decimals=6)

    def objective(self,trial):

        params = {
             "stable_model_update_freq": trial.suggest_float("stable_model_update_freq", 0.10, 0.35, step=0.05),
             "plastic_model_update_freq":trial.suggest_float("plastic_model_update_freq", 0.70, 1.0, step=0.05),
             "stable_model_alpha":trial.suggest_float("stable_model_alpha", 0.10, 0.70, step=0.05),
             "plastic_model_alpha":trial.suggest_float("plastic_model_alpha", 0.80, 1.0, step=0.05),
             "num_syntheticExamplesPerDigit":trial.suggest_int("num_syntheticExamplesPerDigit", 4, 10, step=3),
             #"num_syntheticExamplesPerDigit":trial.suggest_int("num_syntheticExamplesPerDigit", 10, 50, step=10),
          #  "total_epochs": trial.suggest_int("total_epochs",5,20,5),   #don't use num_epochs, it matches with some reserved names and throws error
            # "reg_weight": trial.suggest_float("reg_weight", 0.25, 0.85, step=0.05),
        #    "patience": trial.suggest_int("patience",3,7,2),  # patience has very little impact, and a value of 3 is ideal for most of the cases
        #   "learning_rate":trial.suggest_float("learning_rate",1e-5,1e-4,step=None,log=True),
         #    "inhibit_factor":trial.suggest_float("inhibit_factor",1e-2,3*1e-1,step=None,log=True), # using a log uniform distribution to find the parameter
          #  "rho":trial.suggest_float("rho", 0.5, 3, step=0.5),
          #  "batch_sizeCLS": trial.suggest_int("batch_sizeCLS",4,16,4),
           # "mini_batchGR": trial.suggest_int("mini_batchGR",4,16,4),
        #    "synthetic_images": trial.suggest_int("synthetic_images",40,200,40)


        }
        self.buffer_images = []
        self.buffer_labels = []
        
        total_epochs= 250      #params['total_epochs']
        n_classes=15

        device = "cuda"
        n_experiences=5
        batch_sizeCLS = 4#16#params['batch_sizeCLS']  #64
        mini_batchGR = 8#16#params['mini_batchGR']  #64

        stable_model_update_freq = params['stable_model_update_freq']#0.10#
        plastic_model_update_freq = params['plastic_model_update_freq'] #0.65 #
        stable_model_alpha = params['stable_model_alpha']#0.10#
        plastic_model_alpha = params['plastic_model_alpha']
        reg_weight = 1e-5
        
        learning_rate = 1e-5#params['learning_rate']
        patience = 45 #params['patience']
        clipping=True

        #################### Hyperparameters Generator #########################
        learning_rateGR = 0.0001 #0.001
        batch_sizeGR = 16 #128
        num_epochsGR = 110#250
        patienceGR = 75  # No patience

        input_featureDim = 600
        latent_embedding = 100
            
        # buffer size = num_syntheticExamplesPerDigit * 10
        num_syntheticExamplesPerDigit = params['num_syntheticExamplesPerDigit']#10

        num_originalExamplesPerDigit = 10
        
        scenario_trainVal = self.dataGeneration()     
        
        #getting the scenario
        train_stream = scenario_trainVal.train_stream
        val_stream = scenario_trainVal.test_stream

        ## Initialize CLS model
        cl_strategy = CustomInhibitStrategy(working_model=WorkingModel,modelstable=StableModel,modelplastic=PlasticModel,\
        stable_model_update_freq=stable_model_update_freq,plastic_model_update_freq=plastic_model_update_freq,\
        num_epochs=total_epochs,reg_weight=reg_weight,batch_size=batch_sizeCLS,n_classes=n_classes,
        n_channel=input_featureDim, patience=patience,learning_rate=learning_rate,stable_model_alpha=stable_model_alpha,
        plastic_model_alpha=plastic_model_alpha,mini_batchGR=mini_batchGR,clipping=clipping) #CLS strategy

        ## Generator model
        gen_model = VAE(input_dim=input_featureDim, latent_embedding=latent_embedding, device=device).to(device=device)
        gen_class = Vae_Cls_Generator(num_epochs=num_epochsGR, model=gen_model, device=device, learning_rate=learning_rateGR, 
                                      batch_size=batch_sizeGR, patience=patienceGR, )

        ## Training and Evaluation for Custom Method
        results = []
        exp_numb = 0
        for experience in train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
            print("Training Generator on current experience")
            gen_class.train(experience)
            for digit in experience.classes_in_this_experience:
                temp_img, temp_labels = utility_funcs.buffer_dataGeneration(digit=digit, experience=experience, num_examples=num_syntheticExamplesPerDigit,
                                                device=device,model=gen_model,numbOf_orgExamples=num_originalExamplesPerDigit, batch_size=batch_sizeGR)
                self.buffer_images.append(temp_img)
                self.buffer_labels.append(temp_labels)

            print("Training CL model on current experience")

            cl_strategy.train(experience,buf_inputs=self.buffer_images,buf_labels=self.buffer_labels)    # Comment for running pre trained cl model
            print('Training completed')

            # **********************For sleep******************************** 
            ## Sleep Phase
            if (exp_numb == n_experiences-1):  
                print("Starting offline learning for reorganizing memories")
                cl_strategy.offline_reorganizing(buf_inputs=self.buffer_images,buf_labels=self.buffer_labels,epochs=30,lr_offline=1e-4,offline_batch=32) 
                print("Reorganization done")
            #########################################

            print('Computing accuracy on the whole test set')
            final_accuracy,acc_dict,_,_ = cl_strategy.evaluate(val_stream,validationFlag = False)
            results.append(final_accuracy)
            exp_numb+=1

        y_stable,y_plastic,cls_output = utility_funcs.dataPrepToPlot(acc_dict)

        #Mean after n experiences
        meanStablePred = np.sum(y_stable)/n_experiences    
        meanPlasticPred = np.sum(y_plastic)/n_experiences
        meanClsOutput = np.sum(cls_output)/n_experiences

        #average_scoreStable = np.sum(meanStablePred)/n_experiences

        print(f"The mean value after 5 experinces for stable model is {np.round(meanStablePred,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for stable model is {np.round(meanStablePred,decimals=4)}")

        print(f"The mean value after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")

        print(f"The mean value after 5 experinces for CLS output model is {np.round(meanClsOutput,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for CLS output model is {np.round(meanClsOutput,decimals=4)}")

        #return meanStablePred
        return meanClsOutput


def main():
    
    ##########################################################
    ##### Optuna trainer
    ##########################################################

    hyperparametr_obj = Hyperparametr()
    StableAccuracyPerConfig = hyperparametr_obj.objective # objective function

    study = optuna.create_study(direction="maximize")
    study.optimize(StableAccuracyPerConfig, n_trials=40)

    print("best trial")
    trial_ = study.best_trial
    print(trial_.values)
    print("*"*20)
    print("best parameters")
    print(trial_.params)
    print("*"*20)

    # saving the plots for intermediate values
    optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=["stable_model_update_freq", "plastic_model_update_freq",
                                                                            "stable_model_alpha","plastic_model_alpha","num_syntheticExamplesPerDigit"])
    plt.tight_layout()
    plt.savefig(f"tb_results/optuna_SensorExp.png")

if __name__=="__main__":
    main()
