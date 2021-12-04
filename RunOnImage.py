# Run net on folder of images and display results

import numpy as np
import FCN_NetModel as NET_FCN # The net Class
import torch
import Visuallization as vis
import cv2
import open3d as o3d
import RGBD_To_XYZMAP
#------------------input parameters-------------------------------------------------------------------------------
InputImage=r"Example/Test.jpg" # Input image file
Trained_model_path =  "logs/Defult.torch" # Train model to use

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DisplayXYZPointCloud=True # Show Point cloud

MaxSize=900 # Max image size if bigger resize to smaller
#************************************Masks and XYZ maps to predict********************************************************************************************************
#******************************Create and Load neural net**********************************************************************************************************************

Net=NET_FCN.Net() # Create net and load pretrained
Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.to(device)
Net.eval()
#*********************************Read image and resize*******************************************************************************

Img=cv2.imread(InputImage)
Img=vis.ResizeToMaxSize(Img,MaxSize)

ImBatch=np.expand_dims(Img,axis=0)
###############################Run Net and make prediction###########################################################################
with torch.no_grad():
    LogDepthMap = Net.forward(Images=ImBatch,TrainMode=False) # Run net inference and get prediction
DepthMap=torch.pow(2.718281,LogDepthMap)
#DepthMap=LogDepthMap
#----------------------------Convert Prediction to numpy-------------------------------------------
DepthMap=DepthMap[0][0].data.cpu().numpy()
#-----------------------------Convert Depth to XYZ map-------------------------------------------------------------------
Factor=4000/DepthMap.max()
#***************************
#
# depth_raw= o3d.io.read_image("Example/Depth2.png")
# d1 = np.asarray(depth_raw).astype(np.float32)  # [:, :, ::-1]
#DepthMapReal= cv2.imread("Example/Depth2.png",-1)
#******************************
depth_raw=o3d.cpu.pybind.geometry.Image((DepthMap*Factor).astype(np.uint16))
color_raw=o3d.cpu.pybind.geometry.Image(Img.astype(np.float32))
rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw)
XYZ, ROI, ImGrey, DepthMap = RGBD_To_XYZMAP.RGBD2XYZ(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
Depth=(255*DepthMap/DepthMap.max()).astype(np.uint8)
vis.show(Img,"Image "+InputImage)
vis.show(Depth,"Depth MaxVal = "+str(Depth.max()))
#vis.show(ROI*255, "ROI MaxVal = " + str(ROI.max()))

vis.ShowXYZMap(XYZ, "XYZ Map")
vis.DisplayPointCloud(XYZ,Img,ROI,step=2)