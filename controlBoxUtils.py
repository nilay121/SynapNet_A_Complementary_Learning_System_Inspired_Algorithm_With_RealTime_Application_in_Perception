import math
import numpy as np
import serial
import struct
import time
import random
import pandas as pd
import serial.tools.list_ports as ports

class GripperData:
    def __init__(self):
        # ------------------------ CONSTANTS ------------------------ #
        self.PMAX = 13 # which maps to about 1.4 bars
        self.arduinoPB = None

    # ------------------------ COMMUNICATION ------------------------ #
    def set_communication(self, aportPB):
        try:
            self.arduinoPB = serial.Serial( 
                                port=aportPB,
                                baudrate=115200,
                                timeout=.1 
                                )
            self.arduinoPB.isOpen() # try to open port
            print ("Valve is opened!")
        except IOError: # if port is already opened, close it and open it again and print message
            self.arduinoPB.close()
            self.arduinoPB.open()
            print ("Valve was already open, was closed and opened again!")

    ## Send values to the Pressure Box
    def send_values(self, U1, U2, U3, L1, L2, L3, G, PMAX):
        if U1 > PMAX or U2 > PMAX or U3 > PMAX or L1 > PMAX or L2 > PMAX or L3 > PMAX or G > 1:
            print("Pressure out of range")
            return

        packet = np.array([106,U1,U2,U3,L1,L2,L3,G],dtype = np.uint8)
        
        if self.arduinoPB.isOpen():
            for value in packet : 
                s = struct.pack('!{0}B'.format(len(packet)), *packet)
                self.arduinoPB.write(s)

    def Syn_PressureSenValues(self, p_value, aportPB, aportFS, baudrateFS, time_delay, max_counter, object, counterFSThreshold = 6):
        counter = 0
        PMAX = 13 
        SensorData = []
        counterFS = 0
        counterFSThreshold = counterFSThreshold
        try:
            arduinoFS = serial.Serial(aportFS, baudrate=baudrateFS) # For reading the data from sensor
            #arduinoFS.isOpen()
            arduinoFS.open()
            print("Opening the port for data collection!!")
        except IOError: # if port is already opened, close it and open it again and print message
            arduinoFS.close()
            arduinoFS.open()
            print ("Port was already open, re-initializing!!")        
        self.set_communication(aportPB=aportPB) 
        start_time = time.time()

        while (len(SensorData)<max_counter):   

            if counterFS < counterFSThreshold:
                temp = arduinoFS.readline()
            else:
                try:
                    if counter == 0:
                        time.sleep(1) # Initial Sleep
                        self.send_values(p_value,0,0,0,0,0,0,PMAX)  # To enable value V1 only with 3 bar
                    line = arduinoFS.readline().decode('utf-8').strip()
                    values = [val for val in line.split(',')]
                    # Record the analog pin values
                    if len(values) == 4:
                        a0, a1, a2, a3 = values
                        print(f"S0: {a0}, S1: {a1}, S2: {a2}, S3: {a3}, counter {counter}")
                        SensorData.append([counter, a0, a1, a2, a3, object])
                    else:
                        print("Error!! Values missing from board")
                    time.sleep(time_delay) # 10ms -->100hz
                    counter+=1
                except UnicodeDecodeError:
                    print("UnicodeDecodeError detected!!! Stopped")
                    self.send_values(0,0,0,0,0,0,0,PMAX) # Release the object
                    self.arduinoPB.close()
                    arduinoFS.close()
            counterFS+=1

        ## Close the connection    
        arduinoFS.close()
        self.send_values(0,0,0,0,0,0,0,PMAX) # Release the object
        self.arduinoPB.close()
        print("Valve is closed!!")
        end_time = time.time()
        total_time = (end_time - start_time)
        print(f"Total time taken for one run is {total_time}")
        
        # # ####################### ONLY FOR TEST ####################
        # SensorData1 = np.random.randn(150, 5)
        # labeldummy = [object for i in range(150)]
        # labeldummy = np.array(labeldummy).reshape(150,1)
        # SensorData = np.concatenate((SensorData1, labeldummy), axis=1)
        return SensorData

    def SensorDataCreation(self, SensorData, filename, training):
        SensorValuesDataFrame = pd.DataFrame(SensorData, 
                                             columns=['Iterations', 'SensorVal1', 'SensorVal2', 'SensorVal3', 'SensorVal4', 'object'])
        if training:
            filename_dir = f"dataset/Uk_data/train/{filename}"
            backuData_dir = f"dataset/Known_data/BP_SensorDataUkAdd.csv"
            SensorValuesDataFrame.to_csv(filename_dir, index=False)

            ## Record the values also in another backup csv file for later use maybe
            df = pd.read_csv(backuData_dir,sep=",")
            df = pd.concat([df, SensorValuesDataFrame],axis=0)
            # df.to_csv("dataset/Known_data/BP_SensorDataUkAdd.csv", index=False)
            df.to_csv(backuData_dir, index=False)
        else:
            filename_dir = f"dataset/Uk_data/test/{filename}"
            backuData_dir = f"dataset/Known_data/BP_SensorDataUkAddTestData.csv"
            SensorValuesDataFrame.to_csv(filename_dir, index=False)

            ## Record the values also in another backup csv file for later use maybe
            df = pd.read_csv(backuData_dir,sep=",")
            df = pd.concat([df, SensorValuesDataFrame],axis=0)
            df.to_csv(backuData_dir, index=False)
        print("Dataframe created and saved")
        return filename_dir

    def Load_Add(self, SensorDataToAdd, filename, training):
        temp_df = pd.DataFrame(SensorDataToAdd, columns=['Iterations', 'SensorVal1', 'SensorVal2', 'SensorVal3', 'SensorVal4', 'object'])
        if training:
            filename_dir = f"dataset/Uk_data/train/{filename}"
            backuData_dir = f"dataset/Known_data/BP_SensorDataUkAdd.csv"
            df = pd.read_csv(filename_dir, sep=",")
            df = pd.concat([df, temp_df],axis=0)
            df.to_csv(filename_dir, index=False)
            
            ## Record the values also in another backup csv file for later use maybe
            df_backu = pd.read_csv(backuData_dir, sep=",")
            df_backu = pd.concat([df_backu, temp_df],axis=0)
            df_backu.to_csv("dataset/Known_data/BP_SensorDataUkAdd.csv", index=False)
        else:
            filename_dir = f"dataset/Uk_data/test/{filename}"
            df = pd.read_csv(filename_dir, sep=",")
            df = pd.concat([df, temp_df],axis=0)
            df.to_csv(filename_dir, index=False)
        print("Data Added to the parent dataframe")

    def StartExpGeneration(self, itr, filename, training, aportPB, aportFS, uk_objectName):
        start_time = time.time()
        for i in range(itr):
            print(f"Starting iteration {i}")
            SensorData = self.Syn_PressureSenValues(p_value=10, aportPB = aportPB, aportFS = aportFS, baudrateFS = 9600, 
                                            time_delay = 0.1, max_counter=150, object=uk_objectName) # delay = 0.5
            SensorDataArray = np.array(SensorData)
            if i==0:
                filename_obt = self.SensorDataCreation(SensorData=SensorDataArray, filename=filename, training=training)
            else:
                self.Load_Add(SensorDataToAdd=SensorDataArray, filename=filename, training=training)
            time.sleep(5) # in seconds
        total_time = time.time()-start_time
        print(f"Total time taken for the completet experiment is {total_time} s")   
        return filename_obt     

# if __name__=="__main__":
#     test = GripperData()
#     labelName = "ding"
#     filenameData_dir = test.StartExpGeneration(itr=3, filename=f"train_object{labelName}.csv", aportPB="COM3", aportFS="COM4", uk_objectName = "ball", training=True)
#     print(filenameData_dir)