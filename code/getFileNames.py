import os
import pandas as pd
import json
import cv2

def CSV_300W_LP(data_dir):
    folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    images = []

    for idx, folder in enumerate(folders):
        folder_path = os.path.join(data_dir, folder)
        folder_images = [image[:-4] for image in os.listdir(folder_path) if '.jpg' in image]

        for image in folder_images:
            image_path = os.path.join(folder, image)
            images.append(image_path)
	
    df = pd.DataFrame(images)
    df.to_csv(os.path.join(data_dir, '300W_LP.txt'), header=False, index=False)
	
def CSV_custom(data_dir, data_type, output_dir, padding_perc = 0.0):

    images_dir = os.path.join(data_dir,'images', data_type)
    annotations = os.path.join(data_dir, 'annotations', '%s_400_rigi_essemble.json' %data_type)

    with open(annotations, 'r') as f:
        annon_dict = json.loads(f.read())

    # Initializes variables
    avail_imgs = annon_dict.keys()
    x1_list = []
    x2_list = []
    y1_list = []
    y2_list = []
    roll_list = []
    pitch_list = []
    yaw_list = []
    image_paths = []
    

    # Gets path for all images
    images = [os.path.join(images_dir, image) for image in os.listdir(images_dir) if 'jpg' or 'png' in image]
    for image in images:
        # read image (to determine size later)
        img = cv2.imread(image)

        # gets images Id
        img_id = os.path.basename(image)[:-4].lstrip('0')

        # ensures the image is in the dictionary key
        if not img_id in avail_imgs:
            continue

        for idx, annon in enumerate(annon_dict[img_id].keys()):

            # ensures we have a face detected
            if not annon_dict[img_id][annon]['head_pose_pred']:
                continue
            bbox = annon_dict[img_id][annon]['head_pose_pred']['box']
            x1 = bbox['x1']
            x2 = bbox['x2']
            y1 = bbox['y1']
            y2 = bbox['y2']

            # add padding to face
            upper_y = int(max(0, y1 - (y2 - y1) * padding_perc))
            lower_y = int(min(img.shape[0], y2 + (y2 - y1) * padding_perc))
            left_x = int(max(0, x1 - (x2 - x1) * padding_perc))
            right_x = int(min(img.shape[1], x2 + (x2 - x1) * padding_perc))
            
            # get head pose labels
            roll = annon_dict[img_id][annon]['head_pose_pred']['roll']
            pitch = annon_dict[img_id][annon]['head_pose_pred']['pitch']
            yaw = annon_dict[img_id][annon]['head_pose_pred']['yaw']

            image_paths.append(os.path.basename(image)[:-4])
            x1_list.append(left_x)
            x2_list.append(right_x)
            y1_list.append(upper_y)
            y2_list.append(lower_y)
            
            roll_list.append(roll)
            pitch_list.append(pitch)
            yaw_list.append(yaw)
            
    # saves data in RetinaNet format
    data = {'image_path': image_paths,
            'x1': x1_list, 'x2': x2_list,
            'y1': y1_list, 'y2': y2_list,
            'roll': roll_list, 'pitch': pitch_list,
            'yaw': yaw_list}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, '%s_labels_headpose_essemble.csv' %data_type), index=False, header=True)


def CSV_AFLW2000(data_dir):
    images = [image[:-4] for image in os.listdir(data_dir) if '.jpg' in image]
    
    df = pd.DataFrame(images)
    df.to_csv(os.path.join(data_dir, 'AFLW2000.txt'), header=False, index=False)


def CSV_BIKI(data_dir):
    folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    
    images = []
    
    for idx, folder in enumerate(folders):
        if folder == 'faces':
            continue
        
        folder_path = os.path.join(data_dir, folder)
        
        # Initiating detector and coco annotations
        folder_images = [os.path.join(folder, frame[:-8]) for frame in os.listdir(folder_path) if 'png' in frame]
        for image in folder_images:
            images.append(image)
        
    
    df = pd.DataFrame(images)
    df.to_csv(os.path.join(data_dir, 'BIWI.txt'), header=False, index=False)


def CSV_BIKI_faces(data_dir):
    folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    
    images = []
    
    for idx, folder in enumerate(folders):
        with open(os.path.join(data_dir, folder, 'face_keypoints.json'), 'r') as f:
            folder_dict = json.loads(f.read())
            
            folder_images = folder_dict[folder].keys()
            for image in folder_images:
                image_path = os.path.join(folder, image[:-4])
                images.append(image_path)
    
    df = pd.DataFrame(images)
    df.to_csv(os.path.join(data_dir, 'BIWI_faces.txt'), header=False, index=False)
    
if __name__ == '__main__':
    data_dir = '/home/giancarlo/Documents/HeadPose-test/deep-head-pose/300W-LP/300W_LP'
    #CSV_300W_LP(data_dir)
	
    data_dir = '/home/giancarlo/Documents/custom_dataset_final_results/data'
    data_type = 'train2017'
    output_dir = '/home/giancarlo/Documents/HeadPose-test/deep-head-pose/Custom'
    CSV_custom(data_dir, data_type, output_dir)
    
    data_dir = '/home/giancarlo/Documents/HeadPose-test/deep-head-pose/AFLW2000'
    #CSV_AFLW2000(data_dir)

    data_dir = '/home/giancarlo/Documents/HeadPose-test/deep-head-pose/BIWI/hpdb'
    #CSV_BIKI(data_dir)

    data_dir = '/home/giancarlo/Documents/HeadPose-test/deep-head-pose/BIWI/hpdb/faces'
    #CSV_BIKI_faces(data_dir)