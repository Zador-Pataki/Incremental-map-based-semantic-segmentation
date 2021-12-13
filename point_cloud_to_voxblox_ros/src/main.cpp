#include <iostream>
#include <fstream>
#include <filesystem>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include <string>

#include <pcl/io/ply_io.h>
#include <pcl/pcl_base.h>
#include <pcl/common/transforms.h>

#include <voxblox_ros/tsdf_server.h>
#include <gflags/gflags.h>

#include <voxblox_ros/conversions.h>
#include <voxblox_ros/ros_params.h>
#include <voxblox/core/common.h>
#include <memory>
#include <assert.h>
#include <ctime>

std::vector<std::vector<std::string>> readCSV(std::string dataset,std::string delimiter){
    std::ifstream file(dataset);
    std::vector<std::vector<std::string>> data;

    std::string line = "";
    while(getline(file, line)){
        std::vector<std::string> vec;
        boost::algorithm::split(vec,line,boost::is_any_of(delimiter));
        data.push_back(vec);
    }
    file.close();
    return data;
} //Creates vector of vectors from CSV
Eigen::MatrixXf CSVtoEigen(std::vector<std::vector<std::string>> dataset, int rows, int cols, bool header){
    Eigen::MatrixXf mat(cols,rows);
    for(int i=0;i<rows; i++){
        for(int j=0;j<cols;j++){
            if(header) mat(j,i) = atof(dataset[i+1][j+1].c_str());
            else mat(j,i) = atof(dataset[i][j].c_str());
        }
    }
    return mat.transpose();
} //Converts vector of vectors to Eigen
Eigen::MatrixXf readCSVtoEigen(std::string dataset_name,std::string delimiter, bool header){
    std::vector<std::vector<std::string>> dataset = readCSV(dataset_name,delimiter);
    int rows;
    int cols;
    if(header) {
        rows = dataset.size() - 1;
        cols = dataset[0].size() - 1;
    }
    else{
        rows = dataset.size();
        cols = dataset[0].size();
    }
    Eigen::MatrixXf CameraPoses_ = CSVtoEigen(dataset,rows,cols, header);
    return CameraPoses_;
} //Creats Eigen matrix from CSV

Eigen::MatrixXf get_cameraPose(Eigen::MatrixXf cameraPoses, int view_idx){
    Eigen::MatrixXf cameraPose = cameraPoses.row(view_idx);
    cameraPose = Eigen::Map<Eigen::MatrixXf>(cameraPose.data(), 4,4);
    return cameraPose.transpose();
} //extracts a single camera pose from set of camera poses


int main(int argc, char** argv)
{
    std::string train_or_val = "val";
    ros::init(argc, argv, "pointCLouds");
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, false);
    google::InstallFailureSignalHandler();

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    nh_private.setParam("color_mode","color");
    nh_private.setParam("tsdf_voxel_size",0.08);

    ros::Publisher pub = nh_private.advertise<pcl::PointCloud<pcl::PointXYZ> >("pointCloud", 1, true);

    std::string data_processed_root_path;
    if (train_or_val == "train"){
        data_processed_root_path = "/media/patakiz/Extreme SSD/train_processed/";
    }
    if (train_or_val == "val"){
        data_processed_root_path = "/media/patakiz/Extreme SSD/val_processed/";
    }

    std::string camera_poses_name;
    std::string file_name;
    std::string point_cloud_dir;
    Eigen::MatrixXf cameraPose;
    Eigen::MatrixXf CameraPoses;
    std::string delimiter = ",";

    std::time_t start_time;
    std::time_t current_time;
    start_time = std::time(NULL);


    for (const auto& train_folder : std::filesystem::directory_iterator(data_processed_root_path)){
        int i = 0;
        if(std::filesystem::is_directory(train_folder)) {
            std::cout<< train_folder << std::endl;
            for (const auto &trajectory_dir: std::filesystem::directory_iterator(train_folder)) {

                //------------------------------------------------------------------------------------------------------
                if (i >= 750 && i < 1001){
                    camera_poses_name = trajectory_dir.path().string();
                    camera_poses_name.append("/camera_poses_flat.csv");
                    std::cout << camera_poses_name << std::endl;  //ssssssssssssssssss
                    CameraPoses = readCSVtoEigen(camera_poses_name, delimiter, 1);
                    voxblox::TsdfServer node(nh, nh_private);

                    node.generateMesh();
                    point_cloud_dir = trajectory_dir.path().string();
                    point_cloud_dir.append("/point_cloud_files/point_clouds_classes/point_cloud");
                    std::cout << point_cloud_dir << std::endl;
                    for (int j = 0; j < CameraPoses.rows(); j++) {
                        file_name = point_cloud_dir;
                        file_name.append(std::to_string(j));
                        file_name.append(".ply");
                        cameraPose = get_cameraPose(CameraPoses, j);
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                        pcl::io::loadPLYFile<pcl::PointXYZRGB>(file_name, *cloud);
                        cloud->header.frame_id = "world";

                        auto color_map = std::make_shared<voxblox::GrayscaleColorMap>(); //Aaii

                        voxblox::Pointcloud pointcloud; //Aaiii ...

                        voxblox::Colors colors; //good enough? sema length as voxblox::Pointcloud?
                        voxblox::convertPointcloud(*cloud, color_map, &pointcloud, &colors);
                        voxblox::Transformation T_G_C(
                                static_cast<voxblox::Transformation::TransformationMatrix>(cameraPose)); //Ab  make sure cameraPose appr homogenous mat shap //aaaa
                        node.integratePointcloud(T_G_C, pointcloud, colors);
                        node.getTsdfMapPtr()->getTsdfLayer();

                        voxblox::BlockIndexList updated_blocks;
                        node.getTsdfMapPtr()->getTsdfLayer().getAllAllocatedBlocks(&updated_blocks);
                        using TsdfBlock = voxblox::Block<voxblox::TsdfVoxel>;
                        int number_of_blocks = updated_blocks.size();
                        int number_of_voxels;

                        for (auto &index : updated_blocks) {
                            const TsdfBlock &tsdf_block = node.getTsdfMapPtr()->getTsdfLayer().getBlockByIndex(index);
                            number_of_voxels = tsdf_block.num_voxels();
                        }
                        //std::cout<<number_of_voxels<<"  "<<number_of_blocks<<std::endl<<std::endl;
                        std::vector<std::vector<float>> out_data(number_of_blocks,
                                                                 std::vector<float>(number_of_voxels * 3));
                        if (((j + 1) % 20 == 0) || (j == 299)) {
                            int block_index = 0;
                            for (auto &index : updated_blocks) {
                                //std::cout<<index<<std::endl; // ASK ABOUT THIS! SEEMS TO LOOP OVER THE SAME INDEX
                                TsdfBlock &tsdf_block = node.getTsdfMapPtr()->getTsdfLayerPtr()->getBlockByIndex(index);

                                //node.getTsdfMapPtr()->getTsdfLayerPtr()->getBlockByIndex(index).updated();
                                for (size_t linear_index = 0; linear_index < tsdf_block.num_voxels(); ++linear_index) {
                                    voxblox::TsdfVoxel &tsdf_voxel = tsdf_block.getVoxelByLinearIndex(linear_index);
                                    out_data[block_index][linear_index] = tsdf_voxel.distance;
                                    out_data[block_index][linear_index + number_of_voxels] = tsdf_voxel.weight;
                                    out_data[block_index][linear_index +
                                                          2 * number_of_voxels] = static_cast<int>(tsdf_voxel.color.r);
                                }
                                block_index++;
                            }

                            std::ofstream out;
                            std::string voxblox_data;
                            if (train_or_val == "train") {
                                voxblox_data = "/media/patakiz/Extreme SSD/voxel_data/";
                            }
                            if (train_or_val == "val") {
                                voxblox_data = "/media/patakiz/Extreme SSD/val_voxel_data/";
                            }

                            voxblox_data.append(train_folder.path().filename().string());
                            voxblox_data.append("/");
                            voxblox_data.append(std::to_string(j + 1));
                            boost::filesystem::create_directory(voxblox_data);
                            //mkdir(voxblox_data);
                            voxblox_data.append("/voxblox_data_");
                            voxblox_data.append(trajectory_dir.path().filename().string());
                            voxblox_data.append(".csv");
                            std::cout << voxblox_data << std::endl;

                            out.open(voxblox_data);
                            for (auto &row : out_data) {
                                for (auto col:row) {
                                    out << col << ',';
                                }
                                out << '\n';
                            }
                        }
                    }
                }
                i++;
                std::cout << '(' << i << ',' << "1000)" << std::endl;
                current_time = std::time(NULL);
                std::cout << "Time Passed:" << (1.0 * current_time - 1.0 * start_time) / 3600.0 << std::endl;
            }
            break;
        }
    }
    return 0;
}
// rosrun package_name execut_name
// marsian cubes
