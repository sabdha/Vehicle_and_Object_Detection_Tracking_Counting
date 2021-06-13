# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:07:32 2021

@author: dhany
"""

import tensorflow as tf
import cv2
import numpy as np


def create_category_index(label_path='C:/Users/dhany/models/research/label.txt'):
  f = open(label_path)
  category_index = {}
  for i, val in enumerate(f):
      if i != 0:
        #to get all types of vehicles
        if i > 8:
          break
        else:
          val = val[:-1]
          if val != '???':
              category_index.update({(i-1): {'id': (i-1), 'name': val}})
          print(category_index)
  f.close()
  return category_index





def apply_nms(output_dict, iou_thresh=0.5, score_thresh=0.6):
  q = 90 # no of classes
  num = int(output_dict['num_detections'])
  boxes = np.zeros([1, num, q, 4])
  scores = np.zeros([1, num, q])
  # val = [0]*q
  for i in range(num):
      # indices = np.where(classes == output_dict['detection_classes'][i])[0][0]
      boxes[0, i, output_dict['detection_classes'][i], :] = output_dict['detection_boxes'][i]
      scores[0, i, output_dict['detection_classes'][i]] = output_dict['detection_scores'][i]
  nmsd = tf.image.combined_non_max_suppression(boxes=boxes,
                                                scores=scores,
                                                max_output_size_per_class=num,
                                                max_total_size=num,
                                                iou_threshold=iou_thresh,
                                                score_threshold=score_thresh,
                                                pad_per_class=False,
                                                clip_boxes=False)
  valid = nmsd.valid_detections[0].numpy()
  output_dict = {
                  'detection_boxes' : nmsd.nmsed_boxes[0].numpy()[:valid],
                  'detection_classes' : nmsd.nmsed_classes[0].numpy().astype(np.int64)[:valid],
                  'detection_scores' : nmsd.nmsed_scores[0].numpy()[:valid],
                  }
  return output_dict




def run_inference_for_single_image(input_data, interpreter, output_details, nms, iou_thresh, score_thresh):
  output_dict = {
                   'detection_boxes' : interpreter.get_tensor(output_details[0]['index'])[0],
                   'detection_classes' : interpreter.get_tensor(output_details[1]['index'])[0],
                   'detection_scores' : interpreter.get_tensor(output_details[2]['index'])[0],
                   'num_detections' : interpreter.get_tensor(output_details[3]['index'])[0]
                   }
  
  
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  #print(output_dict['detection_classes'])
  if nms == True:
      output_dict = apply_nms(output_dict, iou_thresh, score_thresh)    
  return output_dict






def show_inference(frame_resized, interpreter, category_index, input_details, output_details, totalFrames):

  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
  frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
  frame_resized = cv2.resize(frame_rgb, (input_width, input_height), cv2.INTER_AREA)
  input_data = np.expand_dims(frame_resized, axis=0)

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  output_dict = run_inference_for_single_image(input_data, interpreter, \
                                               output_details, nms=True, iou_thresh=0.6, score_thresh=0.6)
  boxes = []
  classes = []
  scores = []
  #print(output_dict['detection_classes'])
  global count_dict
#  score_thresh = 0.6
  for i,x in enumerate(output_dict['detection_classes']):
       if output_dict['detection_scores'][i] > 0.6 and x == 2:
         print("category",x)
         classes.append(x)
         boxes.append(output_dict['detection_boxes'][i])
         scores.append(output_dict['detection_scores'][i])  
         #for object count, the score should be more than 54% for accuracy
         #if output_dict['detection_classes'][i] > 0 and output_dict['detection_scores'][i] > 0.6\
         #    and output_dict['detection_scores'][i] < 0.7:
          # count_dict = count_dict + 1
         
           
  boxes = np.array(boxes)
  classes = np.array(classes)
  scores = np.array(scores)

  # check to see if we should write the frame to disk
  return boxes, classes, scores, frame_resized
#  cv2.putText(frame, "Vehicles Detected : "+str(count_dict), (150, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 3)

