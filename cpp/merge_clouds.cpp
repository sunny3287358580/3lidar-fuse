#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "用法: " << argv[0]
                  << " <中间.pcd> <左.pcd> <右.pcd>" << std::endl;
        return -1;
    }

    // 载入原始点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ref(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_left(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_right(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud_ref) == -1 ||
        pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *cloud_left) == -1 ||
        pcl::io::loadPCDFile<pcl::PointXYZ>(argv[3], *cloud_right) == -1) {
        std::cerr << "无法读取 PCD 文件。" << std::endl;
        return -1;
    }

    // ===== 这里填入你标定出来的外参 =====
    Eigen::Matrix4f T_ref_left = Eigen::Matrix4f::Identity();
    T_ref_left <<
        0.998705, -0.00857183,  0.0502153, -2.79488,
        0.000261474, 0.986594,  0.16321,    0.00431993,
       -0.050941,  -0.162984,   0.985316,  -0.384993,
        0, 0, 0, 1;

    Eigen::Matrix4f T_ref_right = Eigen::Matrix4f::Identity();
    T_ref_right <<
        0.999515, -0.0170292,  0.0262958,  2.49478,
        0.00968027, 0.966169,  0.257744,  -0.394543,
       -0.0297949, -0.257363,  0.965858,  -0.462209,
        0, 0, 0, 1;

    // 变换到中间雷达坐标系
    pcl::PointCloud<pcl::PointXYZ>::Ptr left_aligned(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr right_aligned(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::transformPointCloud(*cloud_left,  *left_aligned,  T_ref_left);
    pcl::transformPointCloud(*cloud_right, *right_aligned, T_ref_right);

    // 融合
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged(new pcl::PointCloud<pcl::PointXYZ>);
    *merged = *cloud_ref;
    *merged += *left_aligned;
    *merged += *right_aligned;

    std::cout << "融合前总点数: " << merged->size() << std::endl;

    // 体素降采样，去除重叠区域冗余点（可按需调整 leaf 大小）
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_ds(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(merged);
    sor.setLeafSize(0.05f, 0.05f, 0.05f);   // 5cm 体素
    sor.filter(*merged_ds);

    std::cout << "融合后总点数: " << merged_ds->size() << std::endl;

    pcl::io::savePCDFileBinary("merged.pcd", *merged_ds);
    std::cout << "已保存到 merged.pcd" << std::endl;

    return 0;
}