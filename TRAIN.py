# Train net that predict depth map from image
#...............................Imports..................................................................
import os
import numpy as np
import FCN_NetModel as NET_FCN # The net Class
import torch
import torch.nn as nn
import torch.nn.functional as F
import LossFunctions
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import Visuallization as vis
import SUN3DReader as DepthReader

#https://proceedings.neurips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf
##################################Input paramaters#########################################################################################


#...............Other training paramters..............................................................................

Sun3DDir=r"/media/breakeroftime/2T/Data_zoo/SUN_RGBD/uu/SUNRGBD//" # Input train directory
MinSize=300 # Min image dimension (height or width)
MaxSize=900# Max image dimension (height or width)
MaxPixels=800*800*4# Max pixels in a batch (not in image), reduce to solve out if memory problems
MaxBatchSize=4# Max images in batch

Trained_model_path="" # Path of trained model weights If you want to return to trained model, else if there is no pretrained mode this should be =""
Learning_Rate=1e-5 # intial learning rate
Weight_Decay = 4e-5
TrainedModelWeightDir="logs/" # Output Folder where trained model weight and information will be stored
TrainLossTxtFile="logs/Loss.txt"
if not os.path.exists(TrainedModelWeightDir): os.mkdir(TrainedModelWeightDir)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#=========================Load net weights from previous case if available====================================================================================================================
InitStep=1
if os.path.exists(TrainedModelWeightDir + "/Defult.torch"):
    Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"):
    Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))

####################Create and Initiate net and create optimizer##########################################################################################3

Net=NET_FCN.Net() # Create net and load pretrained

#--------------------if previous model exist load it--------------------------------------------------------------------------------------------
if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.to(device)
#--------------------------------Optimizer--------------------------------------------------------------------------------------------
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer

#----------------------------------------Create reader for data sets--------------------------------------------------------------------------------------------------------------
DepthReader=DepthReader.Reader(MainDir=Sun3DDir,MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)


#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------

if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + "test" + ".torch") # test saving to see the everything is fine

f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
#-------------------Loss Parameters--------------------------------------------------------------------------------
PrevAvgLoss=0 # Average loss in the past (to compare see if loss as been decrease)
AVGLoss=0 # Average loss for each prediction


############################################################################################################################
#..............Start Training loop: Main Training....................................................................
print("Start Training")
for itr in range(InitStep,100000000000): # Main training loop
   #print("------------------------------" , itr , "------------------------------------------------")

    #***************************Reading batch ******************************************************************************
   Imgs, GTDepth, GT_XYZ, ROI = DepthReader.LoadBatch()
    #***************************************************************************************************
   # for i in range(Imgs.shape[0]):
   #      print("DepthRange", GTDepth.max())
   #      GTDepth[i] = (255 * GTDepth[i] / GTDepth[i].max()).astype(np.uint8)
   #      vis.show(Imgs[i])
   #      vis.show(GTDepth[i], "Depth MaxVal = " + str(GTDepth[i].max()))
   #      vis.show(ROI[i] * 255, "ROI MaxVal = " + str(ROI[i].max()))
   #      vis.ShowXYZMap(GT_XYZ[i], "XYZ Map")
   #      vis.DisplayPointCloud(GT_XYZ[i], Imgs[i], ROI[i], step=2)
    #*****************************************************************************
   PrdLogDepth = Net.forward(Images=Imgs) # Run net inference and get prediction
   Net.zero_grad()
 #**************************************Depth Map Loss*************************************************************************************************************************
           #------------------------ROI to prediction size (prediction map is shrink version of the input image)----------------------------------------------------
   ROI = torch.autograd.Variable(torch.from_numpy(ROI).unsqueeze(1).to(device),requires_grad=False) # ROI to torch
   ROI = nn.functional.interpolate(ROI, tuple((PrdLogDepth.shape[2], PrdLogDepth.shape[3])), mode='bilinear',align_corners=False)  # ROI to net output scale
   ROI[ROI < 0.9] = 0  # Resize have lead to some intirmidiate values ignore them
   ROI[ROI > 0.9] = 1  # Resize have lead to some intirmidiate values ignore them

   TGT =  torch.log(torch.from_numpy(GTDepth).to(device).unsqueeze(1) + 0.0001)  ### GT Depth log
   TGT.requires_grad = False
   TGT = nn.functional.interpolate(TGT, tuple((PrdLogDepth.shape[2], PrdLogDepth.shape[3])), mode='bilinear',align_corners=False) # convert to prediction size
   Loss = 5*LossFunctions.DepthLoss(PrdLogDepth, TGT, ROI) # Loss function
#==========================================================================================================================
   # --------------Apply backpropogation-----------------------------------------------------------------------------------

   Loss.backward()  # Backpropogate loss
   optimizer.step()  # Apply gradient descent change to weight
#---------------Calculate Total Loss  and average loss by using the sum of all objects losses----------------------------------------------------------------------------------------------------------

   fr = 1 / np.min([itr - InitStep + 1, 2000])
   AVGLoss=(1 - fr) * AVGLoss + fr * Loss.data.cpu().numpy()




###############################################################################################################################
#===================Display, Save and update learning rate======================================================================================
#########################################################################################################################33
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
   if itr % 300 == 0:# and itr>0: #Save model weight once every 300 steps, temp file
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
   if itr % 60000 == 0 and itr>0: #Save model weight once every 60k steps permenant file
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
   if itr % 20==0: # Display train loss and write to statics file
        txt="\n"+str(itr)

        txt+="\tAverage Loss="+str(AVGLoss)+"  Learning Rate "+str(Learning_Rate)
        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()
# #----------------Update learning rate -------------------------------------------------------------------------------
   if itr%20000==0:
        if PrevAvgLoss==0:
            APrevAvgLoss=AVGLoss
        elif AVGLoss*0.95<PrevAvgLoss: # If average loss havent decrease in the last 20k steps update training loss
            Learning_Rate*=0.9 # Reduce learning rate
            if Learning_Rate<=3e-7: # If learning rate to small increae it
                Learning_Rate=5e-6
            print("Learning Rate="+str(Learning_Rate))
            print("======================================================================================================================")
            optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Create adam optimizer with new learning rate
        AVGLoss=AVGLoss+0.0000000001 # Save current average loss for later comparison
