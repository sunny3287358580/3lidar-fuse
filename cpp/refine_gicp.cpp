// Independent source -> center GICP baseline. All units: meters.
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/gicp.h>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
Cloud::Ptr load(const std::string& path) {
  Cloud::Ptr c(new Cloud), clean(new Cloud);
  if (pcl::io::loadPCDFile(path, *c) < 0) throw std::runtime_error("Cannot read "+path);
  std::vector<int> indices; pcl::removeNaNFromPointCloud(*c, *clean, indices);
  return clean;
}
Cloud::Ptr sample(Cloud::Ptr c, float leaf) {
  if (leaf <= 0) return c;
  Cloud::Ptr d(new Cloud); pcl::VoxelGrid<pcl::PointXYZ> f;
  f.setInputCloud(c); f.setLeafSize(leaf,leaf,leaf); f.filter(*d); return d;
}
int main(int argc, char** argv) {
 try {
  if(argc < 5) {std::cerr << "usage: refine_gicp source.pcd center.pcd initial4x4.txt output4x4.txt [newton|bfgs]\n";return 2;}
  auto src=load(argv[1]), tgt=load(argv[2]);
  Eigen::Matrix4f T; std::ifstream in(argv[3]);
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)if(!(in>>T(i,j)))throw std::runtime_error("Bad initial matrix");
  const float leaves[]={.08f,.04f,.02f,.01f,0.f};
  const double gates[]={.30,.18,.12,.08,.06};
  bool bfgs=argc>5 && std::string(argv[5])=="bfgs";
  for(int level=0;level<5;++level) {
    auto a=sample(src,leaves[level]), b=sample(tgt,leaves[level]);
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> g;
    g.setInputSource(a);g.setInputTarget(b);g.setCorrespondenceRandomness(20);
    g.setMaxCorrespondenceDistance(gates[level]);g.setMaximumIterations(400);
    g.setMaximumOptimizerIterations(100);g.setTransformationEpsilon(1e-9);
    g.setRotationEpsilon(1e-9);g.setTranslationGradientTolerance(1e-8);g.setRotationGradientTolerance(1e-8);
    if(bfgs)g.useBFGS();
    Cloud aligned;g.align(aligned,T);
    auto next=g.getFinalTransformation();
    if(!next.allFinite() || std::abs(next.block<3,3>(0,0).determinant()-1)>1e-3)throw std::runtime_error("Invalid GICP result");
    T=next;std::cout<<"leaf="<<leaves[level]<<" gate="<<gates[level]<<" converged="<<g.hasConverged()<<" fitness="<<g.getFitnessScore(.06)<<"\n"<<std::setprecision(10)<<T<<std::endl;
    std::ofstream out(argv[4]);out<<std::setprecision(12)<<T<<"\n";
  }
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
 return 0;
}
