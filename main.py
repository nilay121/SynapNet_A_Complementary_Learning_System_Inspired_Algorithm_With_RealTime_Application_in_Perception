'''
SynapNet is Brain Inspired Complementary Learning System implementation with a fast Learner (hippocampus), 
a slow learner (Neocortex), lateral Inhibition and a sleep phase for re-organizing the memories.
'''

import torch
import torchvision
import time
import numpy as np
import pandas as pd
from vae_model import VAE
import matplotlib.pyplot as plt
from plasticModel import PlasticModel
from workingModel import WorkingModel 
from stableModel import StableModel
from vae_training import Vae_Cls_Generator
from argparse import ArgumentParser
from controlBoxUtils import GripperData
from cls_inhibition_algo import CustomInhibitStrategy
from avalanche.benchmarks.generators import nc_benchmark
from sklearn.model_selection import train_test_split
from utils import utility_funcs, CustomDatasetForDataLoader
utility_funcs = utility_funcs()

class SynapNet():
    def __init__(self):

        # CLS Model Parameters
        # knownLabelsListInitial = utility_funcs.readList()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 250 
        self.n_classes = 15
        self.n_experiences=5 
        self.batch_sizeCLS = 4
        self.mini_batchGR = 8
        self.offline_batch = 32 

        self.stable_model_update_freq = 0.25
        self.plastic_model_update_freq = 0.85
        self.reg_weight = 1e-5
        self.stable_model_alpha = 0.30000000000000004
        self.plastic_model_alpha = 0.9

        self.patience = 45
        self.learning_rate=1e-5  
        self.clipping = True

        ## Generator Parameters
        self.learning_rateGR = 1e-4
        self.batch_sizeGR = 16 
        self.num_epochsGR = 110
        self.patienceGR = 70  

        self.input_featureDim = 600
        self.latent_embedding = 100

        self.num_syntheticExamplesPerDigit = 7
        self.num_originalExamplesPerDigit = 7

    def singleRun(self, Uk_classExpPhase=False):
        tactileData = pd.read_csv("dataset/Known_data/BP_SensorData.csv", sep=",")
        self.labels = ['solid_square', 'solid_cylinder', 'stapler', 'airpods_case', 'gluestick', 'bottle', 'rectangular_block', 
                       'hollow_cylinder', 'airpods', 'mobile_phone', 'smart_watch', 'air_guage', 'big_cuboid', 'plier', 'ball']      

        modifiedData = []
        modifiedLabels = []
        for i in range(len(self.labels)):
            dataToTransform = np.array(tactileData.loc[tactileData['object'] == self.labels[i], 
                                                       ['SensorVal1', 'SensorVal2', 'SensorVal3', 'SensorVal4']])
            for j in range(0,dataToTransform.shape[0],150):
                tempTrans = dataToTransform[j:j+150].reshape(1,-1)
                modifiedLabels.append(i)
                modifiedData.append(tempTrans)
            print("All transformations done for experiences in train and test!!!")

        modifiedLabels = np.array(modifiedLabels)
        modifiedData = np.array(modifiedData).squeeze(1)

        train_indices, val_indices, _, _ = train_test_split(range(len(modifiedLabels)),modifiedLabels,stratify=modifiedLabels,
                test_size=0.20)
        
        trainData = modifiedData[train_indices]
        trainLabels = modifiedLabels[train_indices]

        testData = modifiedData[val_indices]
        testLabels = modifiedLabels[val_indices]

        X_normalizedTrain = utility_funcs.normalizer(trainData)
        X_normalizedTest = utility_funcs.normalizer(testData)

        X_normalizedTrain = X_normalizedTrain.astype("float32")
        X_normalizedTest = X_normalizedTest.astype("float32")

        trainData = CustomDatasetForDataLoader(data=X_normalizedTrain, targets=trainLabels)
        testData = CustomDatasetForDataLoader(data=X_normalizedTest, targets=testLabels)

        scenarioTrainVal = nc_benchmark(trainData, testData, n_experiences=self.n_experiences, shuffle=False, seed=9, task_labels=False)

        train_stream = scenarioTrainVal.train_stream
        self.test_stream =  scenarioTrainVal.test_stream

        ## Initialize CLS model
        self.cl_strategy = CustomInhibitStrategy(working_model=WorkingModel,modelstable=StableModel,modelplastic=PlasticModel,\
        stable_model_update_freq=self.stable_model_update_freq,plastic_model_update_freq=self.plastic_model_update_freq,
        num_epochs=self.num_epochs,reg_weight=self.reg_weight,batch_size=self.batch_sizeCLS,n_classes=self.n_classes,
        n_channel=self.input_featureDim, patience=self.patience,learning_rate=self.learning_rate,
        plastic_model_alpha=self.plastic_model_alpha, stable_model_alpha=self.stable_model_alpha, 
        mini_batchGR=self.mini_batchGR,clipping=self.clipping) #CLS strategy

        ## Initialize Generator model
        gen_model = VAE(input_dim=self.input_featureDim, latent_embedding=self.latent_embedding, device=self.device).to(device=self.device)
        gen_class = Vae_Cls_Generator(num_epochs=self.num_epochsGR, model=gen_model, device=self.device, learning_rate=self.learning_rateGR, 
                                      batch_size=self.batch_sizeGR, patience=self.patienceGR, )

        ## Train and Evaluate
        print('Starting experiment...')
        self.buffer_images = []
        self.buffer_labels = []
        exp_numb = 0
        for experience in train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
            print("Training Generator on current experience")
            if Uk_classExpPhase==False:
                gen_class.train(experience)

                for digit in experience.classes_in_this_experience:
                    temp_img, temp_labels = utility_funcs.buffer_dataGeneration(digit=digit, experience=experience, num_examples=self.num_syntheticExamplesPerDigit,
                                                    device=self.device,model=gen_model,numbOf_orgExamples=self.num_originalExamplesPerDigit,batch_size=self.batch_sizeGR)
                    self.buffer_images.append(temp_img)
                    self.buffer_labels.append(temp_labels)
                
                print("Generator tarining completed")
                print("Training CL model on current experience")

                ## Train the CL Model
                ## Comment the training once trained on all the experiences
                self.cl_strategy.train(experience, buf_inputs=self.buffer_images, buf_labels=self.buffer_labels)       
                print('Training completed')
            
                ## Sleep Phase
                if (exp_numb == self.n_experiences-1):  
                    print("Starting offline learning for reorganizing memories")
                    stable_model, plastic_model, working_model = self.cl_strategy.offline_reorganizing(buf_inputs=self.buffer_images,
                                                                                                       buf_labels=self.buffer_labels,
                                                                                                       epochs=30,lr_offline=1e-4,
                                                                                                       offline_batch=self.offline_batch) 
                    print("Reorganization done")
            else:
                self.cl_strategy = torch.load("models/cl_strategy.pickle")
            ## Accuracy Computation   
            print('Computing accuracy on the whole test set')
            _, acc_dict, _, _ = self.cl_strategy.evaluate(self.test_stream)
            exp_numb+=1

            # ## Confusion Matrix for each experience
            # print(" Plotting the confusion matrix for each experience ")
            # utility_funcs.ConfusionMatrixPerExp(predictionsForCF_stable= predictionsForCF_stable,predictionsForCF_plastic = predictionsForCF_plastic 
            # , ground_truth = test_stream,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], exp_numb=exp_numb, n_experiences=n_experiences)

        ##Save the result for plots
        y_stable, y_plastic, cls_output = utility_funcs.dataPrepToPlot(acc_dict)
        
        ## Save the buffer, CLS model and VAE model
        if Uk_classExpPhase==False:
            utility_funcs.dumpObj(buffer_images=self.buffer_images, buffer_labels=self.buffer_labels, cl_strategy=self.cl_strategy, 
                                  stable_model=stable_model, plastic_model=plastic_model, working_model=working_model, gen_model=gen_model)

        return y_stable, y_plastic, cls_output, self.n_experiences, self.test_stream, self.n_classes

def main():
    # Acquire the arguments
    parser = ArgumentParser(description="SynapNet Application")
    parser.add_argument("--pseudo_exp", type=bool, required=True, help="Perform pseudo experiment instead of real-time",
                        choices=[True, False])
    parser.add_argument("--Uk_classExpPhase", type=bool, required=True, help="Perform dynamic model expansion for new classes",
                        choices=[True, False])
    parser.add_argument("--num_runs", type=int, required=False, help="Number of runs for intial experience training",
                        default=1)
    args = parser.parse_args()

    Uk_classExpPhase = args.Uk_classExpPhase
    pseudo_exp = args.pseudo_exp
    num_runs = args.num_runs
    counter = 0
    stablePredN = []
    plasticPredN = []
    cls_outputPredN = []
    
    # Call the parent class 
    newExpLearner = SynapNet()
    
    # Control Box, Sensor call
    data_acquisition = GripperData()
    for i in range(num_runs):
        print("*"*10)
        print(f" Starting Repeatation Number {counter} out of {num_runs}")
        print("*"*10)
        ## Make Uk_classExpPhase = False, if you want to train the CLS algorithm from scratch  
        y_stable, y_plastic, cls_output, n_experiences, test_streamExp, n_classes  = newExpLearner.singleRun(Uk_classExpPhase=Uk_classExpPhase) 
        knownLabelsListInitial = utility_funcs.readList()

        stablePredN.append(y_stable)
        plasticPredN.append(y_plastic)
        cls_outputPredN.append(cls_output)
        counter+=1

    #Mean, std after N runs runs for n experinces
    meanStablePred = np.round(np.sum(stablePredN,axis=0)/num_runs,decimals=2)    
    meanPlasticPred = np.round(np.sum(plasticPredN,axis=0)/num_runs,decimals=2)
    meanClsOutput = np.round(np.sum(cls_outputPredN,axis=0)/num_runs, decimals=2)

    stdStablePred = np.round(np.std(stablePredN,axis=0), decimals=4)   
    stdPlasticPred = np.round(np.std(plasticPredN,axis=0), decimals=4)
    stdClsOutput = np.round(np.std(cls_outputPredN,axis=0), decimals=4)

    print(f"The mean accuracy after {n_experiences} experiences for {num_runs} for stable model is {np.sum(meanStablePred)/n_experiences}")
    print(f"The Corresponding std. {n_experiences} experiences for {num_runs} for stable model is {np.sum(stdStablePred)/n_experiences}")

    print(f"The mean accuracy after {n_experiences} experiences for {num_runs} for plastic model is {np.sum(meanPlasticPred)/n_experiences}")
    print(f"The Corresponding std. {n_experiences} experiences for {num_runs} for plastic model is {np.sum(stdPlasticPred)/n_experiences}")

    print(f"The mean accuracy after {n_experiences} experiences for {num_runs} CLS output model is {np.sum(meanClsOutput)/n_experiences}")
    print(f"The Corresponding std. {n_experiences} experiences for {num_runs} CLS output model is {np.sum(stdClsOutput)/n_experiences}")

    utility_funcs.barPlotMeanPred(y_plotPlastic= meanPlasticPred, y_plotStable = meanStablePred, y_clsOutput= meanClsOutput,stdStablePred=stdStablePred,
    stdPlasticPred=stdPlasticPred, stdClsOutput=stdClsOutput, n_experinces = n_experiences)
    
    if Uk_classExpPhase:
        ## For the unknwon data
        print("#"*10,"Starting Test on Unknown object","#"*10)
        newClass = input("Enter the object name: ")
        if newClass == "":
            print("No class name found!!! Re-starting")
            time.sleep(1)
            main()
        else:
            y_stableAkt, y_plasticAkt, cls_outputAkt, knownLabelsList = utility_funcs.OnlineTest(threshold = 0.30, 
                                                                                expTestStream = test_streamExp, 
                                                                                labelsUk_trainTest = newClass, 
                                                                                totalClasses=  n_classes, 
                                                                                num_syntheticExamplesPerDigit = 10, 
                                                                                acquisition_function = data_acquisition, 
                                                                                knownLabelsList=knownLabelsListInitial,
                                                                                customCLSobj = CustomInhibitStrategy, aportFS="COM4", 
                                                                                num_originalExamplesPerDigit = 5, aportPB="COM3", 
                                                                                pseudo_exp=pseudo_exp, train_itr=10, test_itr=3, 
                                                                                in_dim=600)
        print(f"The classes seen so far are {knownLabelsList}")
        utility_funcs.writeList(knownLabelsList)

if __name__=="__main__":
    main()

