import os
import pandas as pd
import json
import cv2

data_dir = '/home/giancarlo/Documents/custom_dataset_final_results/data'
data_type = 'train2017'
annotations_hopenet = os.path.join(data_dir, 'annotations', '%s_400_rigi.json' %data_type)

with open(annotations_hopenet, 'r') as f:
    hopenet_dict = json.loads(f.read())
        
annotations_FSA = os.path.join(data_dir, 'annotations', '%s_400_rigi_FSA.json' %data_type)

with open(annotations_FSA, 'r') as f:
    FSA_dict = json.loads(f.read())

for img_id in FSA_dict.keys():
	for annon_id in FSA_dict[img_id].keys():
		average_yaw = (hopenet_dict[img_id][annon_id]['head_pose_pred']['yaw']+FSA_dict[img_id][annon_id]['head_pose_pred']['yaw'])/2
		average_pitch = (hopenet_dict[img_id][annon_id]['head_pose_pred']['pitch']+FSA_dict[img_id][annon_id]['head_pose_pred']['pitch'])/2
		average_roll = (hopenet_dict[img_id][annon_id]['head_pose_pred']['roll']+FSA_dict[img_id][annon_id]['head_pose_pred']['roll'])/2
		
		FSA_dict[img_id][annon_id]['head_pose_pred']['yaw'] = average_yaw
		FSA_dict[img_id][annon_id]['head_pose_pred']['pitch'] = average_pitch
		FSA_dict[img_id][annon_id]['head_pose_pred']['roll'] = average_roll
		
# save result out to json
r = json.dumps(FSA_dict, indent=4)

with open('train2017_400_rigi_essemble.json', 'w') as f:
	f.write(r)