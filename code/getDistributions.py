import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params


def get_head_pose_charts(data_dir, data_type, save_dir):
    
    annotations = os.path.join(data_dir, 'annotations', '%s_400_rigi_essemble.json' % data_type)
    
    with open(annotations, 'r') as f:
        annon_dict = json.loads(f.read())

    roll_set = []
    pitch_set = []
    yaw_set = []

    for idx, image in enumerate(annon_dict.keys()):
        for annon in annon_dict[image].keys():
            # ensures that there is a face prediction for the annontation
            if not annon_dict[image][annon]['head_pose_pred']:
                continue

            # calculates area
            roll = annon_dict[image][annon]['head_pose_pred']['roll']
            pitch = annon_dict[image][annon]['head_pose_pred']['pitch']
            yaw = annon_dict[image][annon]['head_pose_pred']['yaw']
            roll_set.append(roll)
            pitch_set.append(pitch)
            yaw_set.append(yaw)

    bins = range(-99,99)
    counts, bins = np.histogram(roll_set, bins = bins)

    plt.figure(0)
    plt.hist(roll_set, bins=bins)
    plt.title("Roll Angle Distribution")

    # saving histogram
    plt.savefig(os.path.join(save_dir, 'roll_histogram.jpg'), bbox_inches='tight')

    counts, bins = np.histogram(pitch_set, bins=bins)
    plt.figure(1)
    plt.hist(pitch_set, bins=bins)
    plt.title("Pitch Angle Distribution")

    # saving histogram
    plt.savefig(os.path.join(save_dir, 'pitch_histogram.jpg'), bbox_inches='tight')

    counts, bins = np.histogram(yaw_set, bins=bins)

    plt.figure(2)
    plt.hist(yaw_set, bins=bins)
    plt.title("Yaw Angle Distribution")

    # saving histogram
    plt.savefig(os.path.join(save_dir, 'yaw_histogram.jpg'), bbox_inches='tight')


def get_head_pose_charts_AFLW2000(data_dir):
    
    filename_path = os.path.join(data_dir, 'AFLW2000.txt')
    
    filename_list = get_list_from_filenames(filename_path)

    roll_set = []
    pitch_set = []
    yaw_set = []
    
    for image in filename_list:
        mat_path = os.path.join(data_dir, image + '.mat')
        pose = get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
    
        roll_set.append(roll)
        pitch_set.append(pitch)
        yaw_set.append(yaw)
    
    bins = range(-99, 99)
    counts, bins = np.histogram(roll_set, bins=bins)
    
    plt.figure(0)
    plt.hist(roll_set, bins=bins)
    plt.title("Roll Angle Distribution")
    
    # saving histogram
    plt.savefig(os.path.join(data_dir, 'roll_histogram.jpg'), bbox_inches='tight')
    
    counts, bins = np.histogram(pitch_set, bins=bins)
    plt.figure(1)
    plt.hist(pitch_set, bins=bins)
    plt.title("Pitch Angle Distribution")
    
    # saving histogram
    plt.savefig(os.path.join(data_dir, 'pitch_histogram.jpg'), bbox_inches='tight')
    
    counts, bins = np.histogram(yaw_set, bins=bins)
    
    plt.figure(2)
    plt.hist(yaw_set, bins=bins)
    plt.title("Yaw Angle Distribution")
    
    # saving histogram
    plt.savefig(os.path.join(data_dir, 'yaw_histogram.jpg'), bbox_inches='tight')
    
if __name__ == "__main__":
    data_dir = '/home/giancarlo/Documents/custom_dataset_final_results/data'
    data_type = 'train2017'
    save_dir = '/home/giancarlo/Documents/HeadPose-test/deep-head-pose/Custom'
    #get_head_pose_charts(data_dir, data_type, save_dir)
    
    data_dir = '/home/giancarlo/Documents/HeadPose-test/deep-head-pose/AFLW2000'
    get_head_pose_charts_AFLW2000(data_dir)