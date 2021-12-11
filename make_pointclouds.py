import os
from pathlib import Path

from plyfile import PlyElement, PlyData
from PIL import Image

from pySceneNetRGBD.read_protobuf import photo_path_from_view, instance_path_from_view, depth_path_from_view
import pySceneNetRGBD.scenenet_pb2 as sn
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from matplotlib import pyplot
from pySceneNetRGBD.convert_instance2class import  NYU_WNID_TO_CLASS, NYU_13_CLASSES, colour_code


def normalize(v):
    return v/np.linalg.norm(v)

def position_to_np_array(position,homogenous=False):
    if not homogenous:
        return np.array([position.x,position.y,position.z])
    return np.array([position.x,position.y,position.z,1.0])

def world_to_camera_with_pose(lookat_pose, camera_pose):
    up = np.array([0,1,0])
    R = np.diag(np.ones(4))
    R[2,:3] = normalize(lookat_pose - camera_pose)
    R[0,:3] = normalize(np.cross(R[2,:3],up))
    R[1,:3] = -normalize(np.cross(R[0,:3],R[2,:3]))
    T = np.diag(np.ones(4))
    T[:3,3] = -camera_pose
    P = np.array([[1,0,0,0],[0,0,1,0], [0,-1,0,0],[0,0,0,1]])
    return (R.dot(T))@P

def camera_to_world_with_pose(lookat_pose, camera_pose):
    return np.linalg.inv(world_to_camera_with_pose(lookat_pose, camera_pose))

def write_ply(points, filepath):
    points=points/1000
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    ply_el = PlyElement.describe(vertex, 'vertex')
    target_path, _ = os.path.split(filepath)
    if target_path != '' and not os.path.exists(target_path):
        os.makedirs(target_path)
    PlyData([ply_el]).write(filepath)


def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    points=points/1000
    labels = labels.astype(int) + 1
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert(num_classes>np.max(labels))

    vertex = []
    for i in range(N):
        vertex.append( (points[i,0],points[i,1],points[i,2],labels[i],0,0) )
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el]).write(filename)

def generate_point_cloud(depth_image, h_FoV, v_FoV):
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    cols, rows = np.meshgrid(np.linspace(0, width - 1, num=width), np.linspace(0, height - 1, num=height))
    center_x = width/2
    center_y = height/2
    fx = 320 / (2*np.tan(h_FoV/2))
    fy =240 / (2*np.tan(v_FoV/2))
    x_over_z = (cols - center_x) /fx
    y_over_z = (rows - center_y) /fy
    points_z_ = depth / (1 +x_over_z**2 + y_over_z**2) ** 0.5
    points_x_ = points_z_ * x_over_z
    points_y_ = points_z_ * y_over_z
    return points_x_, points_y_, points_z_

def class_from_instance(instance_path, mapping):
    instance_img = np.asarray(Image.open(instance_path))
    class_img = np.zeros(instance_img.shape)
    for instance, semantic_class in mapping.items():
        class_img[instance_img == instance] = semantic_class
    class_img = Image.fromarray(np.uint8(class_img))
    return np.array(class_img)

def return_true():
   return true


########################################################################################################################
#'pySceneNetRGBD/data/val' : Val dir

horizontal_FoV = np.pi/3 #60 degrees
vertical_FoV = np.pi/4 #40 degrees

data_root_path = '/cluster/work/riner/users/PLR-2021/map-segmentation/scenenet_data/val'
data_processed_root_path = '/cluster/work/riner/users/PLR-2021/map-segmentation/scenenet_data/val_processed'

do_break=False

pc_range = [950, 1001]

for pb in range(1):
    print(pb)
    #protobuf_path = '/cluster/work/riner/users/PLR-2021/map-segmentation/scenenet_data/train_protobufs/scenenet_rgbd_train_' + str(pb)+'.pb'
    protobuf_path = '/cluster/work/riner/users/PLR-2021/map-segmentation/scenenet_data/scenenet_rgbd_val.pb'
    trajectories = sn.Trajectories()


    try:
        with open(protobuf_path,'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(data_root_path))
        print('Please ensure you have copied the pb file to the data directory')



    traj_num = 0
    for traj in tqdm(trajectories.trajectories):
        print(traj_num, traj.render_path)
        if traj_num>=pc_range[0] and traj_num<=pc_range[1]:
            instance_class_map = {}

            for instance in traj.instances:
                instance_type = sn.Instance.InstanceType.Name(instance.instance_type)
                if instance.instance_type != sn.Instance.BACKGROUND:
                    instance_class_map[instance.instance_id] = NYU_WNID_TO_CLASS[instance.semantic_wordnet_id]
        


            camera_poses=[]
            camera_poses_dir = data_processed_root_path +'/'+ traj.render_path
            print(camera_poses_dir + '/camera_poses_flat.csv')
            for i, view in enumerate(traj.views):
                depth_path = depth_path_from_view(traj.render_path,view)
                depth = imageio.imread(depth_path)

                depth = np.asarray(depth)

                points_x, points_y, points_z = generate_point_cloud(depth_image=depth, h_FoV=horizontal_FoV, v_FoV=vertical_FoV)

                points_xyz = np.concatenate((points_x.flatten()[:,np.newaxis],
                                         points_y.flatten()[:,np.newaxis],points_z.flatten()[:,np.newaxis]), axis=1)
                instance_path = instance_path_from_view(traj.render_path,view)

                class_img = class_from_instance(instance_path, instance_class_map)
                class_img_flat = class_img.flatten()
                point_clouds_classes_dir = data_processed_root_path +'/'+ traj.render_path + '/point_cloud_files/point_clouds_classes'
                is_dir = False
                try:Path(point_clouds_classes_dir).mkdir(parents=True, exist_ok=False)
                except: is_dir=True
                if is_dir==True and os.path.isfile(camera_poses_dir + '/camera_poses_flat.csv'): 
                    do_break = False#True
                    #break

                write_ply_color(points_xyz, class_img_flat, point_clouds_classes_dir +'/point_cloud'+str(i)+'.ply')
                lookat = (position_to_np_array(view.shutter_open.lookat) + position_to_np_array(view.shutter_close.lookat))/2
                camera = (position_to_np_array(view.shutter_open.camera) + position_to_np_array(view.shutter_close.camera))/2
                #R = world_to_camera_with_pose(lookat,camera)
                R = camera_to_world_with_pose(lookat,camera)

                camera_poses.append(R.flatten())

            if not do_break:
                camera_poses = np.vstack(camera_poses)
                camera_poses = pd.DataFrame(camera_poses)
            
                try:Path(camera_poses_dir).mkdir(parents=True, exist_ok=False)
                except:pass
                camera_poses.to_csv(camera_poses_dir + '/camera_poses_flat.csv')
                print(camera_poses_dir + '/camera_poses_flat.csv')
            else: 
                do_break=False
        
        traj_num+=1

    break





















