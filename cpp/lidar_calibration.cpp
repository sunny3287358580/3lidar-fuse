// #include <iostream>
// #include <vector>
// #include <string>
// #include <pcl/io/pcd_io.h>
// #include <pcl/point_types.h>
// #include <pcl/registration/gicp.h>          // GICP
// #include <pcl/registration/icp.h>           // ICP（粗配准用）
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/common/transforms.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <Eigen/Dense>

// // 辅助函数：体素降采样
// pcl::PointCloud<pcl::PointXYZ>::Ptr downsample(
//     const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
//     float leaf_size)
// {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::VoxelGrid<pcl::PointXYZ> sor;
//     sor.setInputCloud(cloud);
//     sor.setLeafSize(leaf_size, leaf_size, leaf_size);
//     sor.filter(*filtered);
//     return filtered;
// }

// // 粗配准：标准 ICP，大对应距离，快速拉到接近正确位置
// Eigen::Matrix4f coarseAlign(
//     pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
//     pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
//     const Eigen::Matrix4f& initial_guess,
//     float max_corr_dist)
// {
//     pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
//     icp.setInputSource(source);
//     icp.setInputTarget(target);
//     icp.setMaximumIterations(80);
//     icp.setTransformationEpsilon(1e-8);
//     icp.setEuclideanFitnessEpsilon(1e-8);
//     icp.setMaxCorrespondenceDistance(max_corr_dist);
//     icp.setRANSACOutlierRejectionThreshold(0.05);

//     pcl::PointCloud<pcl::PointXYZ> final_cloud;
//     icp.align(final_cloud, initial_guess);

//     std::cout << "[粗配准] 收敛=" << icp.hasConverged()
//               << ", 适应度=" << icp.getFitnessScore() << std::endl;
//     return icp.getFinalTransformation();
// }

// // 精配准：GICP，小对应距离，逐步收紧
// Eigen::Matrix4f fineAlign(
//     pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
//     pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
//     const Eigen::Matrix4f& initial_guess,
//     float max_corr_dist)
// {
//     pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
//     gicp.setInputSource(source);
//     gicp.setInputTarget(target);
//     gicp.setMaximumIterations(100);
//     gicp.setTransformationEpsilon(1e-10);       // 更严格的收敛条件
//     gicp.setEuclideanFitnessEpsilon(1e-10);
//     gicp.setMaxCorrespondenceDistance(max_corr_dist);
//     gicp.setCorrespondenceRandomness(20);       // 协方差估计邻域
//     gicp.setMaximumOptimizerIterations(128);

//     pcl::PointCloud<pcl::PointXYZ> final_cloud;
//     gicp.align(final_cloud, initial_guess);

//     std::cout << "[精配准-GICP] 收敛=" << gicp.hasConverged()
//               << ", 适应度=" << gicp.getFitnessScore() << std::endl;
//     return gicp.getFinalTransformation();
// }

// // 精度评估：把 source 用 T 变换后，找 target 最近邻，统计距离
// void evaluateResidual(
//     pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
//     pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
//     const Eigen::Matrix4f& T,
//     const std::string& name)
// {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::transformPointCloud(*source, *transformed, T);

//     pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
//     kdtree.setInputCloud(target);

//     std::vector<float> dists;
//     dists.reserve(transformed->size());

//     std::vector<int> idx(1);
//     std::vector<float> sqdist(1);

//     for (const auto& p : transformed->points) {
//         if (kdtree.nearestKSearch(p, 1, idx, sqdist) > 0) {
//             dists.push_back(std::sqrt(sqdist[0]));
//         }
//     }

//     if (dists.empty()) return;

//     std::sort(dists.begin(), dists.end());
//     double sum = 0;
//     for (float d : dists) sum += d;
//     double mean = sum / dists.size();
//     double median = dists[dists.size() / 2];
//     double p90 = dists[static_cast<size_t>(dists.size() * 0.9)];

//     std::cout << "[" << name << "] 残差统计 (米):  "
//               << "mean=" << mean
//               << ", median=" << median
//               << ", p90=" << p90
//               << ", max=" << dists.back() << std::endl;
// }

// // 两阶段标定：粗 ICP -> 精 GICP（对应距离逐步收紧）
// Eigen::Matrix4f calibrate(
//     pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
//     pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
//     const Eigen::Matrix4f& initial_guess,
//     const std::string& name)
// {
//     // 阶段1：粗配准，对应距离 0.5 米
//     Eigen::Matrix4f T1 = coarseAlign(source, target, initial_guess, 0.5f);

//     // 阶段2：GICP 精配准，逐步收紧对应距离
//     Eigen::Matrix4f T2 = fineAlign(source, target, T1, 0.15f);
//     Eigen::Matrix4f T3 = fineAlign(source, target, T2, 0.08f);

//     std::cout << name << " 最终变换矩阵:\n" << T3 << std::endl;
//     evaluateResidual(source, target, T3, name + " 精度");
//     return T3;
// }

// int main(int argc, char** argv) {
//     if (argc != 4) {
//         std::cerr << "用法: " << argv[0]
//                   << " <中间雷达.pcd> <左雷达.pcd> <右雷达.pcd>" << std::endl;
//         return -1;
//     }

//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ref(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_left(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_right(new pcl::PointCloud<pcl::PointXYZ>);

//     if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud_ref) == -1 ||
//         pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *cloud_left) == -1 ||
//         pcl::io::loadPCDFile<pcl::PointXYZ>(argv[3], *cloud_right) == -1) {
//         std::cerr << "无法读取 PCD 文件。" << std::endl;
//         return -1;
//     }

//     std::cout << "原始点数: 中间=" << cloud_ref->size()
//               << ", 左=" << cloud_left->size()
//               << ", 右=" << cloud_right->size() << std::endl;

//     // ===== 关键改进1：用更小的体素（3cm），保留细节 =====
//     float leaf = 0.03f;
//     auto ref_ds   = downsample(cloud_ref, leaf);
//     auto left_ds  = downsample(cloud_left, leaf);
//     auto right_ds = downsample(cloud_right, leaf);

//     std::cout << "降采样(3cm)后点数: 中间=" << ref_ds->size()
//               << ", 左=" << left_ds->size()
//               << ", 右=" << right_ds->size() << std::endl;

//     // 左雷达初值
//     Eigen::Matrix4f init_left = Eigen::Matrix4f::Identity();
//     init_left(0, 3) = -2.54f;

//     // 右雷达初值
//     Eigen::Matrix4f init_right = Eigen::Matrix4f::Identity();
//     init_right(0, 3) = 2.58f;

//     // ===== 关键改进2：两阶段粗->精标定 =====
//     std::cout << "\n===== 左雷达 标定 =====" << std::endl;
//     Eigen::Matrix4f T_ref_left = calibrate(left_ds, ref_ds, init_left, "左雷达");

//     std::cout << "\n===== 右雷达 标定 =====" << std::endl;
//     Eigen::Matrix4f T_ref_right = calibrate(right_ds, ref_ds, init_right, "右雷达");

//     // 保存外参到文件，方便后续融合
//     std::ofstream ofs("extrinsics.txt");
//     ofs << "# 左雷达 -> 中间(Ref)\n" << T_ref_left << "\n\n";
//     ofs << "# 右雷达 -> 中间(Ref)\n" << T_ref_right << "\n";
//     ofs.close();
//     std::cout << "\n外参已保存到 extrinsics.txt" << std::endl;

//     // ===== 融合输出 =====
//     pcl::PointCloud<pcl::PointXYZ>::Ptr left_aligned(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr right_aligned(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::transformPointCloud(*cloud_left,  *left_aligned,  T_ref_left);
//     pcl::transformPointCloud(*cloud_right, *right_aligned, T_ref_right);

//     pcl::PointCloud<pcl::PointXYZ>::Ptr merged(new pcl::PointCloud<pcl::PointXYZ>);
//     *merged = *cloud_ref;
//     *merged += *left_aligned;
//     *merged += *right_aligned;

//     pcl::PointCloud<pcl::PointXYZ>::Ptr merged_ds(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::VoxelGrid<pcl::PointXYZ> sor;
//     sor.setInputCloud(merged);
//     sor.setLeafSize(0.02f, 0.02f, 0.02f);   // 2cm 体素融合
//     sor.filter(*merged_ds);

//     pcl::io::savePCDFileBinary("merged.pcd", *merged_ds);
//     std::cout << "融合点云已保存到 merged.pcd，点数=" << merged_ds->size() << std::endl;

//     return 0;
// }


#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

CloudT::Ptr downsample(const CloudT::Ptr& in, float leaf) {
    CloudT::Ptr out(new CloudT);
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(in);
    vg.setLeafSize(leaf, leaf, leaf);
    vg.filter(*out);
    return out;
}

Eigen::Matrix4f globalFPFH(const CloudT::Ptr& src, const CloudT::Ptr& tgt) {
    // 法线
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    ne.setSearchMethod(tree);
    ne.setKSearch(20);

    pcl::PointCloud<pcl::Normal>::Ptr src_norm(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr tgt_norm(new pcl::PointCloud<pcl::Normal>);
    ne.setInputCloud(src); ne.compute(*src_norm);
    ne.setInputCloud(tgt); ne.compute(*tgt_norm);

    // FPFH
    pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setSearchMethod(tree);
    fpfh.setKSearch(20);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_feat(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr tgt_feat(new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh.setInputCloud(src); fpfh.setInputNormals(src_norm); fpfh.compute(*src_feat);
    fpfh.setInputCloud(tgt); fpfh.setInputNormals(tgt_norm); fpfh.compute(*tgt_feat);

    // RANSAC 全局配准
    pcl::SampleConsensusPrerejective<PointT, PointT, pcl::FPFHSignature33> sac;
    sac.setInputSource(src);
    sac.setSourceFeatures(src_feat);
    sac.setInputTarget(tgt);
    sac.setTargetFeatures(tgt_feat);
    sac.setMaximumIterations(100000);
    sac.setNumberOfSamples(3);
    sac.setCorrespondenceRandomness(5);
    sac.setSimilarityThreshold(0.9f);
    sac.setMaxCorrespondenceDistance(0.5f);
    sac.setInlierFraction(0.15f);

    CloudT out;
    sac.align(out);
    if (!sac.hasConverged()) {
        std::cerr << "FPFH 未收敛！" << std::endl;
    } else {
        std::cout << "FPFH 收敛，适应度=" << sac.getFitnessScore() << std::endl;
    }
    return sac.getFinalTransformation();
}

Eigen::Matrix4f refineGICP(const CloudT::Ptr& src, const CloudT::Ptr& tgt,
                          const Eigen::Matrix4f& init, float max_dist) {
    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
    gicp.setInputSource(src);
    gicp.setInputTarget(tgt);
    gicp.setMaximumIterations(100);
    gicp.setTransformationEpsilon(1e-10);
    gicp.setEuclideanFitnessEpsilon(1e-10);
    gicp.setMaxCorrespondenceDistance(max_dist);
    gicp.setCorrespondenceRandomness(20);

    CloudT out;
    gicp.align(out, init);
    std::cout << "GICP 收敛=" << gicp.hasConverged()
              << ", 适应度=" << gicp.getFitnessScore() << std::endl;
    return gicp.getFinalTransformation();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "用法: " << argv[0]
                  << " <中间.pcd> <左.pcd> <右.pcd>" << std::endl;
        return -1;
    }

    CloudT::Ptr ref(new CloudT), left(new CloudT), right(new CloudT);
    pcl::io::loadPCDFile(argv[1], *ref);
    pcl::io::loadPCDFile(argv[2], *left);
    pcl::io::loadPCDFile(argv[3], *right);

    // FPFH 要求点数不能太多，先降到 5cm
    auto ref_ds   = downsample(ref,   0.05f);
    auto left_ds  = downsample(left,  0.05f);
    auto right_ds = downsample(right, 0.05f);

    std::cout << "\n===== 左雷达全局配准 =====" << std::endl;
    Eigen::Matrix4f T_left_global = globalFPFH(left_ds, ref_ds);
    std::cout << "全局变换:\n" << T_left_global << std::endl;

    std::cout << "\n===== 左雷达 GICP 精配 =====" << std::endl;
    Eigen::Matrix4f T_left = refineGICP(left_ds, ref_ds, T_left_global, 0.15f);
    T_left = refineGICP(left_ds, ref_ds, T_left, 0.08f);   // 再收紧

    std::cout << "\n===== 右雷达全局配准 =====" << std::endl;
    Eigen::Matrix4f T_right_global = globalFPFH(right_ds, ref_ds);
    std::cout << "全局变换:\n" << T_right_global << std::endl;

    std::cout << "\n===== 右雷达 GICP 精配 =====" << std::endl;
    Eigen::Matrix4f T_right = refineGICP(right_ds, ref_ds, T_right_global, 0.15f);
    T_right = refineGICP(right_ds, ref_ds, T_right, 0.08f);

    std::cout << "\n========== 最终外参 ==========" << std::endl;
    std::cout << "左雷达 -> 中间:\n" << T_left << std::endl;
    std::cout << "右雷达 -> 中间:\n" << T_right << std::endl;

    // 融合保存
    CloudT::Ptr left_aligned(new CloudT), right_aligned(new CloudT);
    pcl::transformPointCloud(*left,  *left_aligned,  T_left);
    pcl::transformPointCloud(*right, *right_aligned, T_right);

    CloudT::Ptr merged(new CloudT);
    *merged = *ref;
    *merged += *left_aligned;
    *merged += *right_aligned;

    CloudT::Ptr merged_ds(new CloudT);
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(merged);
    vg.setLeafSize(0.02f, 0.02f, 0.02f);
    vg.filter(*merged_ds);

    pcl::io::savePCDFileBinary("merged.pcd", *merged_ds);
    std::cout << "\n融合点云 saved: merged.pcd, 点数=" << merged_ds->size() << std::endl;

    return 0;
}