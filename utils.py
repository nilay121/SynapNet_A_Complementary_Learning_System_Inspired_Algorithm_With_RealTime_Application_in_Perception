import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import copy
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from vae_training import Vae_Cls_Generator
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from uk_datasetModel import ukDatasetModel

class CustomDatasetForDataLoader(Dataset):
    def __init__(self, data, targets):
        # convet labels to 1 hot
        self.data = data
        self.targets = targets
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.targets[idx]

class utility_funcs:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def buffer_dataGeneration(self, digit, experience,device, batch_size,model,num_examples,numbOf_orgExamples):

        images = []
        synthetic_imgs = []
        labelsForSyntheticImages = []
        batch_size = batch_size
        idx = 0
        constraining_term = 1#0.15
        originalImage_example = 0

        data = experience.dataset
        dataset = DataLoader(data,batch_size=batch_size,shuffle=False)

        for data in dataset:
            x = data[0].to(device)
            y = data[1].cpu().detach().numpy()
            indices_img, = np.where(y==digit)
            for i in indices_img:
                images.append(x[i])
                originalImage_example += 1
                if (originalImage_example == numbOf_orgExamples):
                    break
        encodings_digit = []
        model.eval()
        for i in range(numbOf_orgExamples):
            with torch.no_grad():
                _,mu, sigma = model.encoding_fn(images[i]) #.view(1, 784)
            encodings_digit.append([mu.squeeze(0).squeeze(0).cpu().detach().numpy(),
                                    sigma.squeeze(0).squeeze(0).cpu().detach().numpy()])
        
        encodings_digit = np.array(encodings_digit)
        
        # take average of the mean and sigma for N examples of the same digit
        mean_encodings_digit = encodings_digit.mean(axis=0)
        
        # make the dimension of mu and sigma as its original dimension
        mu = mean_encodings_digit[0]
        mu = torch.as_tensor(mu).unsqueeze(0).unsqueeze(1).to(self.device)

        sigma = mean_encodings_digit[1]
        sigma = torch.as_tensor(sigma).unsqueeze(0).unsqueeze(1).to(self.device)
        
        #Extracting imgaes from latent embedding
        for example in range(num_examples):
            with torch.no_grad():
                epsilon = torch.randn_like(sigma)
                z = mu + (constraining_term*sigma * epsilon)
                out = model.decoder(z).cpu().detach().numpy()
            synthetic_imgs.append(out)
            labelsForSyntheticImages.append(digit)

        synthetic_imgs = np.array(synthetic_imgs)
        return synthetic_imgs, labelsForSyntheticImages

    def toPlotGRImages(self, images_tensor,image_height,image_width,step_size):

        buffer_images = torch.as_tensor(np.array(images_tensor))
        buffer_images = buffer_images.squeeze(2).reshape(-1,1,image_height,image_width)

        print(f"The size of the buffer is {buffer_images.shape[0]}")

        num_images = 20
        color_channels = buffer_images.shape[1]
        high_limit = 10 * step_size

        for i in range(0,high_limit,step_size):
            fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 2.5), sharey=True)
            new_images = buffer_images[i:i+num_images]
            for ax, img in zip(axes, new_images):
                curr_img = img.detach().to(torch.device('cpu'))        

                if color_channels > 1:
                    curr_img = np.transpose(curr_img, (1, 2, 0))
                    ax.imshow(curr_img)
                else:
                    ax.imshow(curr_img.view((image_height, image_width)), cmap='binary')
            fig.savefig(f"synBuffImages/test.png{i/step_size}.png")

    def get_dataBuffer(self, buffer_data, buffer_labels, size,device,transform= None):
        '''
        Getting data from the Generator buffer in a mini batch
        '''
        buffer_inputs = torch.as_tensor(np.array(buffer_data))
        buffer_inputs = buffer_inputs.squeeze(2).reshape(-1, 600,1).to(device)
        buffer_label = torch.as_tensor(np.array(buffer_labels).reshape(-1)).to(device)

        choice = np.random.choice(buffer_inputs.shape[0],size=size, replace=False)

        if transform is None: transform = lambda x: x
        temp_images = torch.stack([transform(ee.cpu()) for ee in buffer_inputs[choice]]).to(self.device)
        temp_labels = buffer_label[choice]
        temp_images = temp_images.reshape(size, 600)
        ## Change the tensor type to Longtensor 
        temp_labels = temp_labels.type('torch.LongTensor').to(device = self.device) 
        return temp_images, temp_labels

    def inputDataTransformation(self, input_data,transform=None):

        if transform is None: transform = lambda x: x
        transformed_images = torch.stack([transform(ee.cpu()) for ee in input_data]).to(self.device)

        return transformed_images

    def benchmarkDataPrep(self, experience, device, synthetic_imgHeight=28, synthetic_imgWidth=28, train_transformBuffer=None, 
                          train_transformInput=None, buffer_data=[],buffer_label=[]):
        
        if len(buffer_data) and len(buffer_label) > 0:
            train_dataset = experience.dataset
            total_dataLength = train_dataset.__len__()
            total_bufferLength = np.array(buffer_label).reshape(-1)

            buffer_inputs, buffer_labels = self.get_dataBuffer(buffer_data=buffer_data,buffer_labels=buffer_label,
            size=len(total_bufferLength),synthetic_imgHeight=synthetic_imgHeight,synthetic_imgWidth=synthetic_imgWidth,device=device,
            transform=train_transformBuffer)

            buffer_inputs = buffer_inputs.cpu().detach().numpy()
            buffer_labels = buffer_labels.cpu().detach().numpy()

            train_data_loader = DataLoader(train_dataset, batch_size=total_dataLength,shuffle=True) 
            for data in  train_data_loader:
                input_dataBT = data[0]
                input_data = self.inputDataTransformation(input_data=input_dataBT,transform=train_transformInput)
                input_data = input_data.cpu().detach().numpy()
                input_labels = data[1].cpu().detach().numpy()
        
            concatenated_inputData = np.concatenate((input_data,buffer_inputs),axis=0)
            concatenated_inputLabels = np.concatenate((input_labels,buffer_labels),axis=0)

            concatenated_inputData = torch.as_tensor(concatenated_inputData)
            concatenated_inputLabels = torch.as_tensor(concatenated_inputLabels)

            newExpDataset = CustomDatasetForDataLoader(data=concatenated_inputData,targets=concatenated_inputLabels)
            return newExpDataset

    def dataPrepToPlot(self, acc_dict):

        y_stable=[]
        y_plastic=[]
        y_working=[]
        cls_output = []

        for i in range(0,len(acc_dict)):
            y_stable.append(np.array(list(acc_dict.values())[i][0].cpu()))
            y_plastic.append(np.array(list(acc_dict.values())[i][1].cpu()))
            y_working.append(np.array(list(acc_dict.values())[i][2].cpu()))
        '''
        The accuracy of the plastic model for the recent experiences are better than the stable model,
        whereas the accuracy of the stable model on the old experiences are better.
        '''
        for outputs in range(len(y_stable)):
            if (y_working[outputs]>y_stable[outputs]): # check if the working model has more accuracy, than the stable one
                y_stable[outputs] = y_working[outputs]
                cls_output.append(y_working[outputs])
            else:
                cls_output.append(y_stable[outputs])

        y_stable = np.array(y_stable)
        y_plastic = np.array(y_plastic)
        cls_output = np.array(cls_output)

        return np.round(y_stable,decimals=2),np.round(y_plastic,decimals=2),np.round(cls_output,decimals=2)
    
    def normalizer(self, data):
        zeroOneRange = ((data-data.min())/(data.max()-data.min()))
        minusOneOneRange = (zeroOneRange - 0.5)/0.5
        return minusOneOneRange
    
    def Uk_dataTrainTest(self, sensorDataFile,  newIdx,labelsUk_trainTest= None, trained_labelsExp=None, training=False):
        '''
        Function to load and transform the data, to make it suitable for training incrementally
        '''
        ExpDataSep = []
        ExpLabelsSep = []
        labelsTT = [labelsUk_trainTest]
        len_labelsExp = len(trained_labelsExp)
        sensorData = pd.read_csv(sensorDataFile)
        trained_labelsExp = trained_labelsExp[0:-1]
        if training == True:
            ## For unknown train data
            for i in range(len(labelsTT)):
                dataToTransform = np.array(sensorData.loc[sensorData['object'] == labelsTT[i], 
                                                          ['SensorVal1', 'SensorVal2', 'SensorVal3', 'SensorVal4']])
                for j in range(0,dataToTransform.shape[0],150):
                    tempTrans = dataToTransform[j:j+150].reshape(1,-1)
                    ExpLabelsSep.append(newIdx) # add the indices of already trained classes
                    ExpDataSep.append(tempTrans)
                print("All transformations done for separate train data!!!")

            trainLabelsUK = np.array(ExpLabelsSep)
            trainDataUK = np.array(ExpDataSep)
            
            X_normalizedTrain_UK = self.normalizer(trainDataUK) 
            X_normalizedTrain_UK = X_normalizedTrain_UK.astype("float32") 

            trainDataUK = CustomDatasetForDataLoader(data=X_normalizedTrain_UK, targets=trainLabelsUK) 
            return trainDataUK
        elif training == False:
            ## For unknown test data
            for i in range(len(labelsTT)):
                dataToTransform = np.array(sensorData.loc[sensorData['object'] == labelsTT[i], 
                                                          ['SensorVal1', 'SensorVal2', 'SensorVal3', 'SensorVal4']])
                for j in range(0,dataToTransform.shape[0],150):
                    tempTrans = dataToTransform[j:j+150].reshape(1,-1)
                    ExpLabelsSep.append(newIdx) # add the indices of already trained classes
                    ExpDataSep.append(tempTrans[0])
                print("All transformations done for separate dummy test data!!!")

            testLabelsUK = np.array(ExpLabelsSep)
            testDataUK = np.array(ExpDataSep)
            
            X_normalizedTrain_UK = self.normalizer(testDataUK) # Unknown data test 
            X_normalizedTrain_UK = X_normalizedTrain_UK.astype("float32") # Unknown data

            testDataUK = CustomDatasetForDataLoader(data=X_normalizedTrain_UK, targets=testLabelsUK) ## Unknwon test data
            return testDataUK 

    # empty contents in a folder
    def empty_folder(self, folder_path):
        files = os.listdir(folder_path)

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(file_path):
                # Remove the file
                os.remove(file_path)
            else:
                # Recursively empty subfolders
                self.empty_folder(file_path) 
        print("Folder Cleared!!")
        
    # Write labels into a file
    def writeList(self, classList):
        with open(r'dataset/Uk_data/UKTestlabels/testLabels.txt', 'w') as fp:
            fp.write('\n'.join(str(item) for item in classList))

    # Read the file with labels
    def readList(self):
        with open(r'dataset/Uk_data/UKTestlabels/testLabels.txt', 'r') as file:
            return [line.strip() for line in file if line.strip()]
                
    def dumpObj(self, buffer_images, buffer_labels, cl_strategy, gen_model, stable_model = None, plastic_model = None, working_model = None):
        self.empty_folder("models")
        self.empty_folder("arrays")
        np.save("arrays/buffer_images", buffer_images) 
        np.save("arrays/buffer_labels", buffer_labels)
        torch.save(cl_strategy, "models/cl_strategy.pickle") # or save the stable and platic model 
        torch.save(gen_model, "models/generator.pickle") 

        if stable_model is not None:
            torch.save(stable_model.state_dict(), "models/stable_model.pth")
            torch.save(plastic_model.state_dict(), "models/plastic_model.pth")
            torch.save(working_model.state_dict(), "models/working_model.pth")
        
    def loadObj(self,):
        buffer_images = np.load("arrays/buffer_images.npy")
        buffer_labels = np.load("arrays/buffer_labels.npy")
        cl_strategy = torch.load("models/cl_strategy.pickle")
        gen_model = torch.load("models/generator.pickle")
        return buffer_images, buffer_labels, cl_strategy, gen_model
    
    def predictionLabels(self, pred, labelsList):
        temp = []
        for i in pred:
            temp.append(labelsList[i])
        print(f"The predictied labels are : {temp}")
    
    # def optimizerStatesChange(self,):
    '''
        We experimented with the optimizer by saving its state before the number of classes exceeded the original fixed threshold,
        but since the working model is only optimized during training and it doesn't require to remember the old task so 
        we removed this function and train the working model from scratch during newThresholdExceedExpCase

    '''
    #     optimizerNew = torch.load('optimizer/optimizer.pth')
    #     # Layer 8 changes 
    #     opt10_ea = optimizerNew['state'][8]['momentum_buffer'].to(self.device)
    #     opt10_rand= torch.rand((1,opt10_ea.shape[1]), device=self.device)
    #     opt10_EaMod = torch.cat((opt10_ea,opt10_rand), axis=0)
    #     optimizerNew['state'][8]['momentum_buffer'] = opt10_EaMod

    #     opt10_eas = optimizerNew['state'][8]['momentum_buffer'].to(self.device)
    #     opt10_rand= torch.rand((1,opt10_eas.shape[1]), device=self.device)
    #     opt10_EasMod = torch.cat((opt10_eas,opt10_rand), axis=0)
    #     optimizerNew['state'][8]['momentum_buffer'] = opt10_EasMod

    #     # Layer 9 changes
    #     opt11_eas = optimizerNew['state'][9]['momentum_buffer'].to(self.device)
    #     opt11_rand= torch.rand((1), device=self.device)
    #     opt11_EasMod = torch.cat((opt11_eas,opt11_rand), axis=0)
    #     optimizerNew['state'][9]['momentum_buffer'] = opt11_EasMod

    #     opt11_ea = optimizerNew['state'][9]['momentum_buffer'].to(self.device)
    #     opt11_rand= torch.rand((1), device=self.device)
    #     opt11_EaMod = torch.cat((opt11_ea, opt11_rand), axis=0)
    #     optimizerNew['state'][9]['momentum_buffer'] = opt11_EaMod

    #     # Save the modified optimizer
    #     torch.save(optimizerNew, 'optimizer/optimizer.pth')
    #     print("#"*5,"Changes made to the optimizer","#"*5)
    
    def addNewClass(self, modelName, in_dim, out_dim):
        path = "models/" + modelName
        newModel = torch.load(path)
        linear1Weight = newModel['model.linear1.weight'].to(self.device)
        linear1bias = newModel['model.linear1.bias'].to(self.device)

        # Random weight and bias
        ## random weight and bias for new class
        torch_randTensor_weight = torch.rand((1,linear1Weight.shape[1]), requires_grad=True, device=self.device)*1e-10 
        torch_randTensor_bias = torch.rand((1), requires_grad=True, device=self.device)*1e-3 

        # modified weight and bias
        modifiedWeightTensor = torch.cat((linear1Weight, torch_randTensor_weight), axis=0)
        modifiedbiasTensor = torch.cat((linear1bias, torch_randTensor_bias), axis=0)

        # overwrite the already stored weight and bias
        newModel['model.linear1.weight'] = modifiedWeightTensor
        newModel['model.linear1.bias'] = modifiedbiasTensor
        # temp_model = torch.load(newModel)
        
        # Model with added new class
        finalModel = ukDatasetModel(input_dim=in_dim, output_dim=out_dim)
        finalModel.load_state_dict(newModel)

        # Saving the new state dictionary for the model
        torch.save(finalModel.state_dict(), path)
        print("Layer modified with the changes!!")
        return finalModel

    def OnlineTest(self, threshold, expTestStream, labelsUk_trainTest, num_syntheticExamplesPerDigit, customCLSobj, 
                   acquisition_function, totalClasses, knownLabelsList, num_originalExamplesPerDigit, in_dim, 
                   aportPB="COM3", aportFS="COM4", train_itr=50, 
                   test_itr=3):
        ## Parameters
        batch_sizeGR = 16
        num_epochs = 110
        learning_rateGR = 1e-4
        classExceeded = False
        trainNewObj = False
        unknownObj = False
        newIdx = 0
        num_syntheticExamplesPerDigit = num_syntheticExamplesPerDigit
        num_originalExamplesPerDigit = num_originalExamplesPerDigit
        len_labelsExp = len(knownLabelsList)
        # Append only the new class to the known labels
        if labelsUk_trainTest in knownLabelsList:
            print("Same object found in the list!!!")
            newIdx = knownLabelsList.index(labelsUk_trainTest)
            print("The object initial index is ", newIdx)
        else:
            knownLabelsList.append(labelsUk_trainTest)
            newIdx = len(knownLabelsList)-1
            unknownObj = True
        print("#"*5, f"Number of known classes is {len(knownLabelsList)} and the initial defined classes is {totalClasses}", "#"*5)
        # Check if the number of classes exceeds the threshold
        if (len(knownLabelsList) <= totalClasses) or (unknownObj==False):
            buffer_images, buffer_labels, cl_strategy, gen_model = self.loadObj()
        elif (len(knownLabelsList) > totalClasses) or (unknownObj==True):
            print("Number of classes exceeded the set threshold!!")
            classExceeded = True
            print("#"*4, "Modifying the network architecture", "#"*4)
            buffer_images, buffer_labels, _, gen_model = self.loadObj()
            sm_modified = self.addNewClass("stable_model.pth", in_dim=in_dim, out_dim=len(knownLabelsList))
            pm_modified = self.addNewClass("plastic_model.pth", in_dim=in_dim, out_dim=len(knownLabelsList))
            wm_modified = self.addNewClass("working_model.pth", in_dim=in_dim, out_dim=len(knownLabelsList))

            cl_strategy = customCLSobj(working_model=wm_modified,modelstable=sm_modified,modelplastic=pm_modified,
                                                stable_model_update_freq=0.30000000000000004,plastic_model_update_freq=0.85,
                                                num_epochs=num_epochs, reg_weight=1e-5, batch_size=4, n_classes=len(knownLabelsList),
                                                n_channel=600, patience=45,learning_rate=1e-5,plastic_model_alpha= 0.9,
                                                stable_model_alpha=0.30000000000000004, mini_batchGR=8, clipping=True, archiChange=True)
        ## Test phase
        # buffer_images, buffer_labels, cl_strategy, gen_model = self.loadObj()
        gen_class = Vae_Cls_Generator(num_epochs=num_epochs, model=gen_model, device=self.device, learning_rate=learning_rateGR, 
                                      batch_size=batch_sizeGR, patience=100 )
        ## Data collection in real time for testing
        testFileName = acquisition_function.StartExpGeneration(itr=test_itr, filename=f"test_object{labelsUk_trainTest}.csv", 
                                                               aportPB=aportPB, aportFS=aportFS, uk_objectName = labelsUk_trainTest, 
                                                               training=False)
        testDataUk = self.Uk_dataTrainTest(sensorDataFile=testFileName, labelsUk_trainTest=labelsUk_trainTest, 
                                           trained_labelsExp=knownLabelsList, training=False, newIdx=newIdx)
        # newClasses = len(pd.Series(testDataUk.targets).unique())
        acc_dict, modelPred, _ = cl_strategy.evaluateUnknown(batch_size=32, test_stream=testDataUk)
        
        ################## test new
        self.predictionLabels(pred= modelPred[0], labelsList= knownLabelsList) 

        y_stable, y_plastic, cls_output = self.dataPrepToPlot(acc_dict)
        
        print(f"Y_stable prediction for unknown data is {y_stable[0]}")
        print(f"Y_plastic prediction for unknown data is {y_plastic[0]}")
        print(f"cls output prediction for unknown data is {cls_output[0]}")

        ## Train
        if y_stable < threshold:
            trainNewObj = True
            print("#"*10,"Collecting real time data from the gripper","#"*10)
            ## Data collection in real time for training
            trainFileName = acquisition_function.StartExpGeneration(itr=train_itr, filename=f"train_object{labelsUk_trainTest}.csv", 
                                                               aportPB=aportPB, aportFS=aportFS, uk_objectName = labelsUk_trainTest, 
                                                               training=True)            
            print("#"*10,"Collection complete!! Loading the data!!","#"*10)
            trainDataUk = self.Uk_dataTrainTest(sensorDataFile=trainFileName, labelsUk_trainTest=labelsUk_trainTest, 
                                                trained_labelsExp=knownLabelsList, training=True, newIdx=newIdx)
            print("#"*10,"Starting training on the unknown data","#"*10)

            scenarioTestUK = nc_benchmark(trainDataUk, trainDataUk, n_experiences=1, shuffle=False, seed=9, task_labels=False)
            train_stream = scenarioTestUK.train_stream
            for exp in train_stream:
                print("#"*10," Training the Generator ","#"*10)
                gen_class.train(exp)
                for digit in exp.classes_in_this_experience:
                    temp_img, temp_labels = self.buffer_dataGeneration(digit=digit, experience=exp, num_examples=num_syntheticExamplesPerDigit,
                                                    device=self.device, model=gen_model, numbOf_orgExamples=num_originalExamplesPerDigit,
                                                    batch_size=batch_sizeGR)
                    buffer_images = buffer_images.reshape(-1, 1, 600, 1)
                    buffer_labels = buffer_labels.reshape(-1)

                    buffer_images = np.concatenate((buffer_images, temp_img), axis=0)
                    buffer_labels = np.concatenate((buffer_labels, temp_labels), axis=0)

                print("Training the CLS algorithm ......>>>>>")
                _, stableModel, plasticModel, workingModel = cl_strategy.train(exp, buf_inputs = buffer_images, buf_labels = buffer_labels)

        ## Test again on th unknown data
        acc_dict, modelPred, _ = cl_strategy.evaluateUnknown(batch_size = 32, test_stream = testDataUk)
        y_stableUk, y_plasticUk, cls_outputUk = self.dataPrepToPlot(acc_dict)
        print(f"Y_stable for unknown data is {y_stableUk[0]}")
        print(f"Y_plastic for unknown data is {y_plasticUk[0]}")
        print(f"cls output for unknown data is {cls_outputUk[0]}")
        self.predictionLabels(pred= modelPred[0], labelsList= knownLabelsList)
        print("#"*10,"Testing on the learned experiences","#"*10)

        ## Test on the learned experiences
        _, acc_dict, _, _ = cl_strategy.evaluate(expTestStream)
        y_stableAkt, y_plasticAkt, cls_outputAkt = self.dataPrepToPlot(acc_dict)

        y_stableAkt = np.concatenate((y_stableAkt, np.round(y_stableUk, decimals=2)), axis=0)
        y_plasticAkt = np.concatenate((y_plasticAkt, np.round(y_plasticUk, decimals=2)), axis=0)
        cls_outputAkt = np.concatenate((cls_outputAkt, np.round(cls_outputUk, decimals=2)), axis=0)

        ## Save the latest model and buffer
        if (classExceeded and trainNewObj) == True:
            self.dumpObj(buffer_images=buffer_images, buffer_labels=buffer_labels,  cl_strategy=cl_strategy, gen_model=gen_model, 
                        stable_model=sm_modified, plastic_model = pm_modified, working_model = wm_modified)
            print("Architecture updated!!")
        elif trainNewObj == True:
            self.dumpObj(buffer_images=buffer_images, buffer_labels=buffer_labels,  cl_strategy=cl_strategy, gen_model=gen_model,
                         stable_model = stableModel, plastic_model = plasticModel, working_model = workingModel)
            print("Model updated!!!")
        return y_stableAkt, y_plasticAkt, cls_outputAkt, knownLabelsList          

    
    def barPlotMeanPred(self, y_plotPlastic,y_plotStable,y_clsOutput,stdStablePred,stdPlasticPred,stdClsOutput,n_experinces):
        N = n_experinces + 1
        ind = np.arange(N)
        width = 0.25
        fig, ax = plt.subplots()

        cls_avgOutputMean = np.round(np.sum(y_clsOutput)/n_experinces,decimals=2)
        cls_avgOutputstd = np.round(np.sum(stdClsOutput)/n_experinces,decimals=2)

        y_plotPlastic = np.insert(y_plotPlastic,obj=n_experinces,values=0)
        stdPlasticPred = np.insert(stdPlasticPred,obj=n_experinces,values=0)

        y_plotStable = np.insert(y_plotStable,obj=n_experinces,values=cls_avgOutputMean)
        stdStablePred = np.insert(stdStablePred,obj=n_experinces,values=cls_avgOutputstd)
        
        bar_plastic = ax.bar(ind, y_plotPlastic, width, color = 'r',label="Plastic Model",yerr=stdPlasticPred)
        bar_stable = ax.bar(ind+width, y_plotStable, width, color='g',label="Stable Model",yerr=stdStablePred)

        ax.axvline(x=4.7,ymin=0,ymax=np.max(y_plotPlastic),color='black', linestyle='dotted', linewidth=2.5)
        
        ax.bar_label(bar_plastic, padding=3)
        ax.bar_label(bar_stable, padding=3)
        
        ax.set_title("Object Classification")
        ax.set_xlabel("Experiences & Models")
        ax.set_ylabel("Accuarcy")
        ax.set_xticks(ind+width,["exp1","exp2","exp3","exp4","exp5","Avg Output"])
        ax.legend((bar_plastic, bar_stable), ('Plastic Model', 'Stable Model'),loc=0)
        fig.tight_layout()
        plt.show()
        plt.savefig(f"pics/tactile/SynapNetApplication{N}exp.png")
    
    def barPlotMeanPredUK(self, y_plotPlastic,y_plotStable,y_clsOutput,n_experinces):
        N = n_experinces + 1
        ind = np.arange(N)
        width = 0.25
        fig, ax = plt.subplots()

        cls_avgOutputMean = np.round(np.sum(y_clsOutput)/n_experinces,decimals=2)

        y_plotPlastic = np.insert(y_plotPlastic,obj=n_experinces,values=0)

        y_plotStable = np.insert(y_plotStable,obj=n_experinces,values=cls_avgOutputMean)
        
        bar_plastic = ax.bar(ind, y_plotPlastic, width, color = 'r',label="Plastic Model")
        bar_stable = ax.bar(ind+width, y_plotStable, width, color='g',label="Stable Model")

        ax.axvline(x=4.7,ymin=0,ymax=np.max(y_plotPlastic),color='black', linestyle='dotted', linewidth=2.5)
        
        ax.bar_label(bar_plastic, padding=3)
        ax.bar_label(bar_stable, padding=3)
        
        ax.set_title("Object Classification")
        ax.set_xlabel("Experiences & Models")
        ax.set_ylabel("Accuarcy")
        ax.set_xticks(ind+width,["exp1","exp2","exp3","exp4","exp5","Avg Output"])
        ax.legend((bar_plastic, bar_stable), ('Plastic Model', 'Stable Model'),loc=0)
        fig.tight_layout()
        plt.savefig(f"pics/tactile/SynapNetApplication{N-1}exp.png")
        plt.show()

    def ConfusionMatrixPerExp(self, predictionsForCF_stable, predictionsForCF_plastic, ground_truth, labels, exp_numb, n_experiences):

        # Extract the ground truth from the experiences
        org_class = []
        exp=0
        last_expLength = 0
        for experiences in ground_truth:
            eval_dataset = experiences.dataset
            total_dataLength = eval_dataset.__len__()
            eval_data_loader = DataLoader(eval_dataset,batch_size=total_dataLength,shuffle=False)
            for data in  eval_data_loader:
                eval_dataLabels = data[1]
            org_class.append(eval_dataLabels.detach().numpy())
            exp+=1
            if exp == (n_experiences):
                last_expLength = eval_dataset.__len__()

        ## Flatten the array 

        predictionsForCF_stable = np.array(predictionsForCF_stable,dtype="object")
        itr_stable = predictionsForCF_stable.shape[0]
        predictionsForCF_stableFalttened = []
        for i in range(itr_stable):
            itr2 = len(predictionsForCF_stable[i])
            for j in range(itr2):
                predictionsForCF_stableFalttened.append(predictionsForCF_stable[i][j])

        predictionsForCF_plastic = np.array(predictionsForCF_plastic,dtype="object")
        itr_plastic = predictionsForCF_plastic.shape[0]
        predictionsForCF_plasticFalttened = []
        for i in range(itr_plastic):
            itr2 = len(predictionsForCF_plastic[i])
            for j in range(itr2):
                predictionsForCF_plasticFalttened.append(predictionsForCF_plastic[i][j])


        org_class = np.array(org_class,dtype="object")
        itr_ground = org_class.shape[0]
        org_classFlattened = []
        for i in range(itr_ground):
            itr2 = len(org_class[i])
            for j in range(itr2):
                org_classFlattened.append(org_class[i][j])

        predictionsForCF_stableFalttened = np.array(predictionsForCF_stableFalttened)
        predictionsForCF_plasticFalttened = np.array(predictionsForCF_plasticFalttened)
        org_classFlattened = np.array(org_classFlattened)
        
        ## Comaprision between stable model and plastic model
        if (exp_numb==n_experiences):
            temp_stablePred = predictionsForCF_stableFalttened[0:(len(predictionsForCF_stableFalttened)-last_expLength)]
            temp_plasticPred = predictionsForCF_plasticFalttened[(len(predictionsForCF_stableFalttened)-last_expLength):]
            cls_output = np.concatenate((temp_stablePred,temp_plasticPred))
        else:
            cls_output = predictionsForCF_stableFalttened
        
        cls_output = np.array(cls_output)
        
        cd_matrix = confusion_matrix(y_true = org_classFlattened, y_pred = cls_output, labels = labels)
        cf_plot = ConfusionMatrixDisplay(cd_matrix,display_labels=labels)
        cf_plot.plot(cmap="BuGn",include_values=False)
        plt.show()
        #plt.savefig(f"confusionMatrix/confusionMatrix{exp_numb}experienceBuffer500.png")