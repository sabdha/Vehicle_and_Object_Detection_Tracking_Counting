# Vehicle_and_Object_Detection_Tracking_Counting 
  
##Open CV, MobileNetSSD detector, Centroid Tracking
  
  1. vehicledetection_mobile.py  
  2. centroids_tracking.py  
  3. trackingobjects.py  
  4. Output File: Opencv_Output.gif
    
### Project Description:  
A Vehicle and people detection, tracking, and counting project, which is a modified version of https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/, which currently tracks only people. This project has various use cases, such as monitoring the road traffic and people at public places.

### Object Detection and tracking:  
 The MobileNetSSD detector is used for the detection of objects. SSD or Single Shot Detector needs only a single shot to detect the objects in an image. It is comparatively faster than the double shot detector like R-CNN. The DNN, which is the MobileNet, makes it compactable to run in mobiles. From the object detection part, we will get the bounding boxes for each detected object in the image.

The tracking part uses the Centroid algorithm given in https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/. The centroid of the bounding boxes is calculated by the centroid tracker algorithm using the coordinates of bounding boxes.  
  
  
 ## Fine Tuning the model:
 
 The vehicles in the video move very fast. So we need to decrease the fps. This can be done by increasing the cv2.waitKey.
 Other parameters to be adjusted are maxdistance, maxdisappeared and the total number of frames to be skipped in between tracking and detection.
 This helps to increase the speed. 
   
  Here is the output for detection and tracking done with OpenCV MobileNet SSD detector and Centroid tracking algorithm. 
     
  ![hippo](https://github.com/sabdha/Vehicle_and_Object_Detection_Tracking_Counting/blob/main/Opencv_Output.gif)    
  Input image from:[Link](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/input_images_and_videos/vehicle_survaillance.mp4)
  
  # Vehicle Detection, Tracking and Counting with Tensorflowlite and OpenCV centroid tracking algorithm  
    
  1. Vehicle_tracking_tensorflowlite.py  
  2. centroids_tracking.py
  3. object_DetectionInfer.py      
  4. trackingobjects.py    
  5. Output File: Tensorflowlite_Output.gif  
    
  I integrated tensorflowlite detection model with the centroid tracking algorithm to see how it works.
  MobileNet SSD model pre-trained on the COCO dataset in Python is used here. The code in this [blogpost](https://towardsdatascience.com/using-tensorflow-lite-for-object-detection-2a0283f94aed) is referred and mostly used to create the tensorflow part.
  ## Non Maximum Suppression  
 Non Maximum Suppression is performed on the retreived bounding boxes and returns the predictions left after the threshold selecting the boxes with highest probability of detection. Later these bounding boxes with best scores were passed to the centroid tracking algorithms to obtain the centroids, their direction and count.
  
  Result:
    
  ![hippo](https://github.com/sabdha/Vehicle_and_Object_Detection_Tracking_Counting/blob/main/Tensorflowlite_Output.gif)  
    
  Input image from:[Link](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/input_images_and_videos/vehicle_survaillance.mp4)
    
  ## Results
  There were many false positives. The confidence has to be adjusted accordingly to obtain the best results.
  
  
  
 
 



