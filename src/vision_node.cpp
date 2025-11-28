#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "vision_msgs/msg/detection3_d.hpp"
#include "vision_msgs/msg/object_hypothesis.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "std_msgs/msg/header.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
/*
#include "referee_pkg/msg/multi_object.hpp"
#include "referee_pkg/msg/object.hpp"
*/
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <chrono>

class VisionNode : public rclcpp::Node
{
public:
    VisionNode() : Node("competition_vision_node")
    {
        // 参数配置
        declare_parameters();
        update_parameters();
        
        // 通信设置
        setup_communications();
        
        // 初始化
        initialize_components();
        
        RCLCPP_INFO(this->get_logger(), "竞赛视觉节点启动完成 - 支持Level 1,2,3");
    }

private:
    // 数据结构
    struct ArmorDetection {
        cv::Rect bbox;
        std::string color;
        std::string shape;
        double confidence;
        int track_id = -1;
        
        std::string get_class_id() const {
            return color + "_" + shape;
        }
    };
    
    struct TrackedTarget {
        int id;
        cv::Rect bbox;
        std::string color;
        std::string shape;
        double confidence;
        geometry_msgs::msg::Point position;
        rclcpp::Time last_seen;
        int age = 0;
        bool confirmed = false;
        
        // Level 3 基础: 速度信息
        geometry_msgs::msg::Point velocity;
        std::vector<geometry_msgs::msg::Point> position_history;
    };

    void declare_parameters()
    {
        // Level 1: 基础检测参数
        this->declare_parameter<double>("vision.detection.min_confidence", 0.6);
        this->declare_parameter<int>("vision.detection.min_armor_area", 100);
        this->declare_parameter<int>("vision.detection.max_armor_area", 5000);
        
        // Level 1: 相机标定参数 (与shooter_node配合)
        this->declare_parameter<std::vector<double>>("vision.camera.matrix", 
            std::vector<double>{800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0});
        this->declare_parameter<std::vector<double>>("vision.camera.dist_coeffs",
            std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});
        
        // Level 1: 物理尺寸
        this->declare_parameter<double>("vision.target.armor_width", 0.13);
        this->declare_parameter<double>("vision.target.armor_height", 0.055);
        
        // Level 2: 颜色阈值 (与shooter_node的颜色权重对应)
        this->declare_parameter<int>("vision.colors.red.h_low1", 0);
        this->declare_parameter<int>("vision.colors.red.s_low1", 120);
        this->declare_parameter<int>("vision.colors.red.v_low1", 70);
        this->declare_parameter<int>("vision.colors.red.h_high1", 10);
        this->declare_parameter<int>("vision.colors.red.s_high1", 255);
        this->declare_parameter<int>("vision.colors.red.v_high1", 255);
        this->declare_parameter<int>("vision.colors.red.h_low2", 170);
        this->declare_parameter<int>("vision.colors.red.s_low2", 120);
        this->declare_parameter<int>("vision.colors.red.v_low2", 70);
        this->declare_parameter<int>("vision.colors.red.h_high2", 180);
        this->declare_parameter<int>("vision.colors.red.s_high2", 255);
        this->declare_parameter<int>("vision.colors.red.v_high2", 255);
        
        this->declare_parameter<int>("vision.colors.blue.h_low", 100);
        this->declare_parameter<int>("vision.colors.blue.s_low", 120);
        this->declare_parameter<int>("vision.colors.blue.v_low", 70);
        this->declare_parameter<int>("vision.colors.blue.h_high", 130);
        this->declare_parameter<int>("vision.colors.blue.s_high", 255);
        this->declare_parameter<int>("vision.colors.blue.v_high", 255);
        
        // Level 2: 跟踪参数
        this->declare_parameter<double>("vision.tracking.iou_threshold", 0.3);
        this->declare_parameter<double>("vision.tracking.max_age", 2.0);
        this->declare_parameter<int>("vision.tracking.min_hits", 3);
        
        // Level 3 基础: 预测参数
        this->declare_parameter<double>("vision.prediction.velocity_smoothing", 0.3);
        this->declare_parameter<int>("vision.prediction.history_size", 5);
        
        // 图像处理参数
        this->declare_parameter<int>("vision.processing.gaussian_kernel", 5);
        this->declare_parameter<double>("vision.processing.gaussian_sigma", 1.0);
        this->declare_parameter<int>("vision.processing.median_kernel", 3);
        
        // 调试参数
        this->declare_parameter<bool>("vision.debug.show_windows", true);
        this->declare_parameter<bool>("vision.debug.log_detections", true);
    }

    void update_parameters()
    {
        // Level 1 参数
        min_confidence_ = this->get_parameter("vision.detection.min_confidence").as_double();
        min_armor_area_ = this->get_parameter("vision.detection.min_armor_area").as_int();
        max_armor_area_ = this->get_parameter("vision.detection.max_armor_area").as_int();
        
        // 相机参数
        auto cam_matrix = this->get_parameter("vision.camera.matrix").as_double_array();
        auto dist_coeffs = this->get_parameter("vision.camera.dist_coeffs").as_double_array();
        
        camera_matrix_ = (cv::Mat_<double>(3,3) << 
            cam_matrix[0], cam_matrix[1], cam_matrix[2],
            cam_matrix[3], cam_matrix[4], cam_matrix[5],
            cam_matrix[6], cam_matrix[7], cam_matrix[8]);
        
        dist_coeffs_ = (cv::Mat_<double>(5,1) << 
            dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3], dist_coeffs[4]);
        
        armor_width_ = this->get_parameter("vision.target.armor_width").as_double();
        armor_height_ = this->get_parameter("vision.target.armor_height").as_double();
        
        // Level 2 颜色阈值
        red_lower1_ = cv::Scalar(
            this->get_parameter("vision.colors.red.h_low1").as_int(),
            this->get_parameter("vision.colors.red.s_low1").as_int(),
            this->get_parameter("vision.colors.red.v_low1").as_int()
        );
        red_upper1_ = cv::Scalar(
            this->get_parameter("vision.colors.red.h_high1").as_int(),
            this->get_parameter("vision.colors.red.s_high1").as_int(),
            this->get_parameter("vision.colors.red.v_high1").as_int()
        );
        red_lower2_ = cv::Scalar(
            this->get_parameter("vision.colors.red.h_low2").as_int(),
            this->get_parameter("vision.colors.red.s_low2").as_int(),
            this->get_parameter("vision.colors.red.v_low2").as_int()
        );
        red_upper2_ = cv::Scalar(
            this->get_parameter("vision.colors.red.h_high2").as_int(),
            this->get_parameter("vision.colors.red.s_high2").as_int(),
            this->get_parameter("vision.colors.red.v_high2").as_int()
        );
        
        blue_lower_ = cv::Scalar(
            this->get_parameter("vision.colors.blue.h_low").as_int(),
            this->get_parameter("vision.colors.blue.s_low").as_int(),
            this->get_parameter("vision.colors.blue.v_low").as_int()
        );
        blue_upper_ = cv::Scalar(
            this->get_parameter("vision.colors.blue.h_high").as_int(),
            this->get_parameter("vision.colors.blue.s_high").as_int(),
            this->get_parameter("vision.colors.blue.v_high").as_int()
        );

        // Level 2 跟踪参数
        iou_threshold_ = this->get_parameter("vision.tracking.iou_threshold").as_double();
        max_age_ = this->get_parameter("vision.tracking.max_age").as_double();
        min_hits_ = this->get_parameter("vision.tracking.min_hits").as_int();

        // Level 3 基础 参数
        velocity_smoothing_ = this->get_parameter("vision.prediction.velocity_smoothing").as_double();
        history_size_ = this->get_parameter("vision.prediction.history_size").as_int();

        // 图像处理参数
        gaussian_kernel_ = this->get_parameter("vision.processing.gaussian_kernel").as_int();
        gaussian_sigma_ = this->get_parameter("vision.processing.gaussian_sigma").as_double();
        median_kernel_ = this->get_parameter("vision.processing.median_kernel").as_int();

        // 调试参数
        show_windows_ = this->get_parameter("vision.debug.show_windows").as_bool();
        log_detections_ = this->get_parameter("vision.debug.log_detections").as_bool();
    }

    void setup_communications()
    {
        // 图像订阅
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&VisionNode::image_callback, this, std::placeholders::_1));

        // 检测结果发布 - 与shooter_node完全兼容
        detection_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(
            "/detections", 10);
/*
        // 裁判系统话题发布器
        referee_pub_ = this->create_publisher<referee_pkg::msg::MultiObject>(
            "/referee/multi_object", 10);
*/            
        // 状态监控
        status_timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&VisionNode::status_callback, this));
    }

    void initialize_components()
    {
        if (show_windows_) {
            cv::namedWindow("Competition Vision", cv::WINDOW_AUTOSIZE);
            cv::namedWindow("Binary Mask", cv::WINDOW_AUTOSIZE);
        }
        morph_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    }

    // 核心处理流程
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            frame_size_ = frame.size();
            
            // Level 1 + Level 2 处理流程
            auto detections = detect_armors(frame);
            auto tracked_detections = track_targets(detections);
            auto final_detections = calculate_3d_positions(tracked_detections);
            
            // 发布结果
            publish_detections(final_detections, msg->header);

            // 发布到裁判系统
//            publish_to_referee(final_detections, msg->header);
            
            // 显示结果
            if (show_windows_) {
                display_results(frame, final_detections);
            }
            
            // 性能监控
            log_performance(start_time, final_detections.size());
            
        } 
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "处理异常: %s", e.what());
        }
    }

    // Level 1: 多颜色装甲板检测
    std::vector<ArmorDetection> detect_armors(const cv::Mat& frame)
    {
        std::vector<ArmorDetection> all_detections;
        
        // 图像预处理
        cv::Mat filtered = apply_filters(frame);
        cv::Mat hsv;
        cv::cvtColor(filtered, hsv, cv::COLOR_BGR2HSV);
        
        // 检测红色和蓝色目标
        auto red_detections = detect_single_color(hsv, "red");
        auto blue_detections = detect_single_color(hsv, "blue");
        
        all_detections.insert(all_detections.end(), red_detections.begin(), red_detections.end());
        all_detections.insert(all_detections.end(), blue_detections.begin(), blue_detections.end());
        
        return all_detections;
    }

    cv::Mat apply_filters(const cv::Mat& frame)
    {
        cv::Mat filtered;
        
        // 高斯滤波
        if (gaussian_kernel_ > 0) {
            cv::GaussianBlur(frame, filtered, cv::Size(gaussian_kernel_, gaussian_kernel_), gaussian_sigma_);
        } else {
            frame.copyTo(filtered);
        }
        
        // 中值滤波
        if (median_kernel_ > 0) {
            cv::medianBlur(filtered, filtered, median_kernel_);
        }
        
        return filtered;
    }

    std::vector<ArmorDetection> detect_single_color(const cv::Mat& hsv, const std::string& color)
    {
        std::vector<ArmorDetection> detections;
        cv::Mat color_mask;
        
        if (color == "red") {
            cv::Mat mask1, mask2;
            cv::inRange(hsv, red_lower1_, red_upper1_, mask1);
            cv::inRange(hsv, red_lower2_, red_upper2_, mask2);
            cv::bitwise_or(mask1, mask2, color_mask);
        } else if (color == "blue") {
            cv::inRange(hsv, blue_lower_, blue_upper_, color_mask);
        } else {
            return detections;
        }
        
        // 显示二值化掩码
        if (show_windows_) {
            cv::imshow("Binary Mask", color_mask);
        }
        
        // 形态学处理
        cv::morphologyEx(color_mask, color_mask, cv::MORPH_CLOSE, morph_kernel_);
        cv::morphologyEx(color_mask, color_mask, cv::MORPH_OPEN, morph_kernel_);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(color_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < min_armor_area_ || area > max_armor_area_) continue;
            
            // Level 1: 形状分类
            std::string shape = classify_shape(contour);
            if (shape == "unknown") continue;
            
            ArmorDetection detection;
            detection.bbox = cv::boundingRect(contour);
            detection.color = color;
            detection.shape = shape;
            detection.confidence = calculate_confidence(contour, shape);
            
            if (detection.confidence >= min_confidence_) {
                detections.push_back(detection);
            }
        }
        
        return detections;
    }

    // Level 1: 形状分类 (与shooter_node的形状权重对应)
    std::string classify_shape(const std::vector<cv::Point>& contour)
    {
        std::vector<cv::Point> approx;
        double epsilon = 0.02 * cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx, epsilon, true);
        
        int vertices = approx.size();
        if (vertices == 3) return "triangle";
        if (vertices == 4) {
            cv::Rect rect = cv::boundingRect(contour);
            double aspect_ratio = static_cast<double>(rect.width) / rect.height;
            return (aspect_ratio > 0.8 && aspect_ratio < 1.2) ? "square" : "rectangle";
        }
        if (vertices >= 8) return "circle";
        
        return "unknown";
    }

    // Level 1: 置信度计算
    double calculate_confidence(const std::vector<cv::Point>& contour, const std::string& shape)
    {
        double area = cv::contourArea(contour);
        cv::Rect rect = cv::boundingRect(contour);
        double rect_area = rect.width * rect.height;
        
        // 基础置信度
        double fill_ratio = area / rect_area;
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double hull_area = cv::contourArea(hull);
        double convexity = hull_area > 0 ? area / hull_area : 0;
        
        double confidence = (fill_ratio * 0.6 + convexity * 0.4);
        
        // 根据形状调整置信度
        if (shape == "rectangle" || shape == "square") {
            // 矩形目标应该具有较高的填充度
            confidence *= std::min(1.0, fill_ratio * 1.2);
        }
        
        return std::min(1.0, confidence);
    }

    // Level 2: 目标跟踪
    std::vector<ArmorDetection> track_targets(const std::vector<ArmorDetection>& detections)
    {
        auto current_time = this->now();
        std::vector<ArmorDetection> tracked_detections;
        
        // 为每个检测寻找最佳匹配的跟踪目标
        for (const auto& detection : detections) {
            int best_match_id = -1;
            double best_iou = iou_threshold_;
            
            for (auto& [id, target] : tracked_targets_) {
                double iou = calculate_iou(detection.bbox, target.bbox);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_match_id = id;
                }
            }
            
            if (best_match_id != -1) {
                // 更新现有目标
                auto& target = tracked_targets_[best_match_id];
                update_tracked_target(target, detection, current_time);
                
                ArmorDetection tracked = detection;
                tracked.track_id = best_match_id;
                tracked_detections.push_back(tracked);
            } else {
                // 创建新目标
                create_new_target(detection, current_time, tracked_detections);
            }
        }
        
        // 清理丢失的目标
        cleanup_lost_targets(current_time);
        
        return tracked_detections;
    }

    void update_tracked_target(TrackedTarget& target, const ArmorDetection& detection, const rclcpp::Time& current_time)
    {
        // 更新基础信息
        target.bbox = detection.bbox;
        target.color = detection.color;
        target.shape = detection.shape;
        target.confidence = detection.confidence;
        target.last_seen = current_time;
        target.age++;
        
        // Level 3 基础: 更新位置历史
        update_position_history(target);
        
        // 确认目标 (达到最小命中次数)
        if (target.age >= min_hits_ && !target.confirmed) {
            target.confirmed = true;
            RCLCPP_DEBUG(this->get_logger(), "目标 %d 已确认", target.id);
        }
    }

    void create_new_target(const ArmorDetection& detection, const rclcpp::Time& current_time, 
                        std::vector<ArmorDetection>& tracked_detections)
    {
        TrackedTarget new_target;
        new_target.id = next_track_id_++;
        new_target.bbox = detection.bbox;
        new_target.color = detection.color;
        new_target.shape = detection.shape;
        new_target.confidence = detection.confidence;
        new_target.last_seen = current_time;
        new_target.age = 1;
        new_target.confirmed = false;
        
        tracked_targets_[new_target.id] = new_target;
        
        ArmorDetection tracked = detection;
        tracked.track_id = new_target.id;
        tracked_detections.push_back(tracked);
        
        RCLCPP_DEBUG(this->get_logger(), "创建新目标 %d: %s", new_target.id, detection.get_class_id().c_str());
    }

    // Level 3 基础: 更新位置历史和速度
    void update_position_history(TrackedTarget& target)
    {
        // 计算当前3D位置
        geometry_msgs::msg::Point current_position;
        precise_3d_localization(target.bbox, current_position);
        target.position = current_position;
        
        // 保存位置历史
        target.position_history.push_back(current_position);
        if (target.position_history.size() > static_cast<size_t>(history_size_)) {
            target.position_history.erase(target.position_history.begin());
        }
        
        // 计算速度 (需要至少2个位置点)
        if (target.position_history.size() >= 2) {
            auto& prev_pos = target.position_history[target.position_history.size() - 2];
            auto& curr_pos = target.position_history.back();
            
            double dt = 0.05; // 假设50ms帧率
            
            geometry_msgs::msg::Point instant_velocity;
            instant_velocity.x = (curr_pos.x - prev_pos.x) / dt;
            instant_velocity.y = (curr_pos.y - prev_pos.y) / dt;
            instant_velocity.z = (curr_pos.z - prev_pos.z) / dt;
            
            // 平滑滤波
            target.velocity.x = target.velocity.x * (1 - velocity_smoothing_) + instant_velocity.x * velocity_smoothing_;
            target.velocity.y = target.velocity.y * (1 - velocity_smoothing_) + instant_velocity.y * velocity_smoothing_;
            target.velocity.z = target.velocity.z * (1 - velocity_smoothing_) + instant_velocity.z * velocity_smoothing_;
        }
    }

    // Level 1: 精确3D定位
    std::vector<ArmorDetection> calculate_3d_positions(const std::vector<ArmorDetection>& detections)
    {
        std::vector<ArmorDetection> result = detections;
        
        for (auto& detection : result) {
            if (detection.track_id != -1) {
                auto& target = tracked_targets_[detection.track_id];
                precise_3d_localization(detection.bbox, target.position);
            }
        }
        
        return result;
    }

    void precise_3d_localization(const cv::Rect& bbox, geometry_msgs::msg::Point& position)
    {
        // 使用PnP算法进行精确3D定位
        std::vector<cv::Point2f> image_points = {
            cv::Point2f(bbox.x, bbox.y),
            cv::Point2f(bbox.x + bbox.width, bbox.y),
            cv::Point2f(bbox.x + bbox.width, bbox.y + bbox.height),
            cv::Point2f(bbox.x, bbox.y + bbox.height)
        };
        
        std::vector<cv::Point3f> object_points = {
            cv::Point3f(-armor_width_/2, -armor_height_/2, 0),
            cv::Point3f(armor_width_/2, -armor_height_/2, 0),
            cv::Point3f(armor_width_/2, armor_height_/2, 0),
            cv::Point3f(-armor_width_/2, armor_height_/2, 0)
        };
        
        cv::Mat rvec, tvec;
        try {
            cv::solvePnP(object_points, image_points, camera_matrix_, dist_coeffs_, rvec, tvec);
            
            position.x = tvec.at<double>(0);
            position.y = tvec.at<double>(1);
            position.z = tvec.at<double>(2);
        } catch (const cv::Exception& e) {
            RCLCPP_WARN(this->get_logger(), "PnP求解失败,使用相似三角形方法");
            fallback_3d_localization(bbox, position);
        }
    }

    void fallback_3d_localization(const cv::Rect& bbox, geometry_msgs::msg::Point& position)
    {
        double fx = camera_matrix_.at<double>(0,0);
        double fy = camera_matrix_.at<double>(1,1);
        
        position.z = (armor_width_ * fx) / bbox.width;
        
        double center_x = bbox.x + bbox.width / 2.0;
        double center_y = bbox.y + bbox.height / 2.0;
        double image_center_x = frame_size_.width / 2.0;
        double image_center_y = frame_size_.height / 2.0;
        
        position.x = (center_x - image_center_x) * position.z / fx;
        position.y = (center_y - image_center_y) * position.z / fy;
    }

    void publish_detections(const std::vector<ArmorDetection>& detections, 
                       const std_msgs::msg::Header& header)
    {
        vision_msgs::msg::Detection3DArray detection_array;
        detection_array.header = header;
        detection_array.header.frame_id = "camera_frame";
        
        for (const auto& detection : detections) {
            if (detection.track_id == -1) continue;
            
            auto& target = tracked_targets_[detection.track_id];
            
            // 只发布已确认的目标
            if (!target.confirmed) continue;
            
            vision_msgs::msg::Detection3D detection_3d;
            
            // 3D位置和尺寸
            detection_3d.bbox.center.position = target.position;
            detection_3d.bbox.size.x = armor_width_;
            detection_3d.bbox.size.y = armor_height_;
            detection_3d.bbox.size.z = 0.05;
            
            // 识别结果 - 与shooter_node的class_id格式完全匹配
            vision_msgs::msg::ObjectHypothesisWithPose result;
            result.hypothesis.class_id = detection.get_class_id();  // "red_rectangle" 等
            result.hypothesis.score = detection.confidence;
            result.pose.pose.position = target.position;
            result.pose.pose.orientation = geometry_msgs::msg::Quaternion();

            detection_3d.results.push_back(result);
            
            detection_array.detections.push_back(detection_3d);
            
            if (log_detections_) {
                RCLCPP_DEBUG(this->get_logger(), "发布目标: %s, 位置: (%.3f, %.3f, %.3f), 速度: (%.2f, %.2f, %.2f)",
                            result.hypothesis.class_id.c_str(),
                            target.position.x, target.position.y, target.position.z,
                            target.velocity.x, target.velocity.y, target.velocity.z);
            }
        }
        
        detection_pub_->publish(detection_array);
    }
/*
    // 发布到裁判系统
    void publish_to_referee(const std::vector<ArmorDetection>& detections, const std_msgs::msg::Header& header)
    {
        referee_pkg::msg::MultiObject multi_obj;
        multi_obj.header = header;  // 使用图像时间戳
        
        for (const auto& detection : detections) {
            if (detection.track_id == -1) continue;
            
            referee_pkg::msg::Object obj;
            
            // 设置目标类型：颜色_形状
            obj.target_type = detection.get_class_id();
            
            // 设置四个角点坐标
            geometry_msgs::msg::Point pt1, pt2, pt3, pt4;
            
            // 从检测框计算四个角点
            pt1.x = detection.bbox.x;
            pt1.y = detection.bbox.y + detection.bbox.height;
            pt1.z = 0;
            
            pt2.x = detection.bbox.x + detection.bbox.width;
            pt2.y = detection.bbox.y + detection.bbox.height;
            pt2.z = 0;
            
            pt3.x = detection.bbox.x + detection.bbox.width;
            pt3.y = detection.bbox.y;
            pt3.z = 0;
            
            pt4.x = detection.bbox.x;
            pt4.y = detection.bbox.y;
            pt4.z = 0;
            
            obj.corners.push_back(pt1);
            obj.corners.push_back(pt2);
            obj.corners.push_back(pt3);
            obj.corners.push_back(pt4);
            
            multi_obj.objects.push_back(obj);
        }
        
        multi_obj.num_objects = multi_obj.objects.size();
        
        // 发布到裁判系统指定话题
        referee_pub_->publish(multi_obj);
        
        if (log_detections_) {
            RCLCPP_DEBUG(this->get_logger(), "发布裁判目标: %zu 个", multi_obj.num_objects);
        }
    }
*/
    void display_results(cv::Mat& frame, const std::vector<ArmorDetection>& detections)
    {
        for (const auto& detection : detections) {
            if (detection.track_id == -1) continue;
            
            auto& target = tracked_targets_[detection.track_id];
            cv::Scalar color = (detection.color == "red") ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
            
            // 绘制边界框
            cv::rectangle(frame, detection.bbox, color, 2);
            
            // 绘制标签 (包含跟踪ID和置信度)
            std::string label = detection.get_class_id() + " ID:" + 
                            std::to_string(detection.track_id) + " " +
                            std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
            
            cv::putText(frame, label,
                    cv::Point(detection.bbox.x, detection.bbox.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            
            // Level 3 基础: 显示速度向量
            if (cv::norm(cv::Point2f(target.velocity.x, target.velocity.y)) > 0.1) {
                cv::Point center(detection.bbox.x + detection.bbox.width/2,
                            detection.bbox.y + detection.bbox.height/2);
                cv::Point velocity_end = center + cv::Point(target.velocity.x * 20, target.velocity.y * 20);
                cv::arrowedLine(frame, center, velocity_end, cv::Scalar(0, 255, 0), 2);
            }
        }
        
        cv::imshow("Competition Vision", frame);
        cv::waitKey(1);
    }

    // 工具函数
    double calculate_iou(const cv::Rect& a, const cv::Rect& b)
    {
        cv::Rect intersection = a & b;
        if (intersection.area() == 0) return 0.0;
        
        cv::Rect union_ = a | b;
        return static_cast<double>(intersection.area()) / union_.area();
    }
    
    void cleanup_lost_targets(const rclcpp::Time& current_time)
    {
        for (auto it = tracked_targets_.begin(); it != tracked_targets_.end(); ) {
            if ((current_time - it->second.last_seen).seconds() > max_age_) {
                RCLCPP_DEBUG(this->get_logger(), "移除丢失目标 %d", it->first);
                it = tracked_targets_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    void log_performance(const std::chrono::high_resolution_clock::time_point& start, size_t detection_count)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start);
        
        static int frame_count = 0;
        if (++frame_count % 60 == 0) {
            RCLCPP_INFO(this->get_logger(), "性能: %zums/帧, 检测目标: %zu, 跟踪目标: %zu", 
                       duration.count(), detection_count, tracked_targets_.size());
        }
    }
    
    void status_callback()
    {
        int confirmed_targets = 0;
        for (const auto& [id, target] : tracked_targets_) {
            if (target.confirmed) confirmed_targets++;
        }
        
        RCLCPP_DEBUG(this->get_logger(), "状态: 总目标=%zu, 已确认=%d", 
                    tracked_targets_.size(), confirmed_targets);
    }

    // 成员变量
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub_;
    rclcpp::TimerBase::SharedPtr status_timer_;

//    rclcpp::Publisher<referee_pkg::msg::MultiObject>::SharedPtr referee_pub_;
    
    // 参数
    double min_confidence_;
    int min_armor_area_, max_armor_area_;
    cv::Mat camera_matrix_, dist_coeffs_;
    double armor_width_, armor_height_;
    
    // 颜色检测
    cv::Scalar red_lower1_, red_upper1_, red_lower2_, red_upper2_;
    cv::Scalar blue_lower_, blue_upper_;
    
    // 跟踪系统
    std::map<int, TrackedTarget> tracked_targets_;
    int next_track_id_ = 0;
    double iou_threshold_, max_age_;
    int min_hits_;
    
    // Level 3 基础
    double velocity_smoothing_;
    int history_size_;
    
    // 图像处理
    int gaussian_kernel_, median_kernel_;
    double gaussian_sigma_;
    cv::Mat morph_kernel_;
    
    // 状态
    cv::Size frame_size_;
    bool show_windows_, log_detections_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
