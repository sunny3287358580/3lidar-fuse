// diag_init.cpp
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>

Eigen::Matrix4f makeInit(float x, float y, float z,
                         float yaw_deg, float pitch_deg, float roll_deg) {
    auto d2r = [](float d) { return d * M_PI / 180.0f; };
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    Eigen::AngleAxisf ry(d2r(yaw_deg),   Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rx(d2r(pitch_deg), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf rz(d2r(roll_deg),  Eigen::Vector3f::UnitZ());
    T.block<3,3>(0,0) = (ry * rx * rz).toRotationMatrix();
    T(0,3) = x; T(1,3) = y; T(2,3) = z;
    return T;
}

int main() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr ref(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr left(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr right(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::io::loadPCDFile("center.pcd", *ref);
    pcl::io::loadPCDFile("left.pcd",   *left);
    pcl::io::loadPCDFile("right.pcd",  *right);

    // ★★★ 先只调这两个矩阵 ★★★
    // 参数含义：x, y, z, yaw, pitch, roll（单位：米 / 度）
    Eigen::Matrix4f T_left  = makeInit(-2.58f, +0.17f, 0.0f,   +30.0f, -20.0f, 0.0f);
    Eigen::Matrix4f T_right = makeInit(+2.54f, +0.15f, 0.0f,   -30.0f, -20.0f, 0.0f);

    std::cout << "T_left:\n"  << T_left  << std::endl;
    std::cout << "T_right:\n" << T_right << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr l2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr r2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*left,  *l2, T_left);
    pcl::transformPointCloud(*right, *r2, T_right);

    // 分别保存，方便在 CloudCompare 里染色叠加
    pcl::io::savePCDFileBinary("diag_center.pcd", *ref);
    pcl::io::savePCDFileBinary("diag_left.pcd",   *l2);
    pcl::io::savePCDFileBinary("diag_right.pcd",  *r2);

    std::cout << "已保存 diag_center.pcd / diag_left.pcd / diag_right.pcd" << std::endl;
    return 0;
}