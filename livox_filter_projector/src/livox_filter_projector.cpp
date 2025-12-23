#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <cmath>
#include <vector>
#include <limits>

class LivoxProcessor {
public:
    LivoxProcessor(ros::NodeHandle& nh) {
        sub_ = nh.subscribe("/livox/lidar", 1, &LivoxProcessor::cloudCallback, this);
        pub_ = nh.advertise<std_msgs::Float32MultiArray>("/lidar_projected", 1);
    }

private:
    ros::Subscriber sub_;
    ros::Publisher pub_;

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        // 转换为PCL点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // Step 1: 过滤 z > 1.0 m 的点
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& pt : cloud->points) {
            if (pt.z <= 1.0) {
                cloud_filtered_z->points.push_back(pt);
            }
            if(pt.z >=0.2) {
                cloud_filtered_z->points.push_back(pt);
            }
        }

        // Step 2: 体素滤波
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud_filtered_z);
        voxel.setLeafSize(0.05f, 0.05f, 0.05f); // 5cm分辨率
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel(new pcl::PointCloud<pcl::PointXYZ>);
        voxel.filter(*cloud_voxel);

        // Step 3: 极坐标映射（359维）
        const int num_bins = 359;
        std::vector<float> min_dist(num_bins, std::numeric_limits<float>::infinity());

        for (const auto& pt : cloud_voxel->points) {
            float r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
            if (r < 0.001) continue;

            float theta = std::atan2(pt.y, pt.x) * 180.0 / M_PI;
            if (theta < 0) theta += 360.0f;

            int bin = static_cast<int>(theta) % num_bins;

            // ±2° 范围内更新最小距离
            for (int d = -2; d <= 2; ++d) {
                int idx = (bin + d + num_bins) % num_bins;
                min_dist[idx] = std::min(min_dist[idx], r);
            }
        }

        // Step 4: 输出结果
        std_msgs::Float32MultiArray output;
        output.data.resize(num_bins);
        for (int i = 0; i < num_bins; ++i) {
            if (std::isinf(min_dist[i])) {
                output.data[i] = 5.0; // 若无障碍，设置默认距离5m
            } else {
                output.data[i] = std::min(min_dist[i], 5.0f); // 限制最大5m
            }
        }

        pub_.publish(output);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "livox_filter_projector");
    ros::NodeHandle nh;

    LivoxProcessor processor(nh);

    ROS_INFO("Livox pointcloud processor started.");
    ros::spin();
    return 0;
}
