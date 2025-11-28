#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <cmath>
#include <optional>
#include <algorithm>

class ShooterNode : public rclcpp::Node
{
public:
    ShooterNode() : Node("level3_shooter_node")
    {
        // 参数声明
        declare_parameters();
        // 订阅发布
        setup_communication();
        // 初始化状态
        initialize_state();
        
        RCLCPP_INFO(this->get_logger(), "击打节点启动");
    }

private:
    // 成员变量
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr detection_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr control_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr fire_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // 状态变量
    double current_pan_;
    double current_tilt_;
    int shot_count_;
    int hit_count_;
    rclcpp::Time last_detection_time_;
    rclcpp::Time cooldown_end_time_;
    std::optional<vision_msgs::msg::Detection3D> current_target_;
    
    // Level 3 专用参数
    double projectile_speed_;
    double gravity_;
    double aim_threshold_;
    double min_confidence_;
    double max_pan_speed_;
    double max_tilt_speed_;
    double cooldown_duration_;
    double velocity_smoothing_;
    double prediction_time_;
    bool enable_prediction_;
    bool debug_mode_;
    
    // Level 3 动态预测专用
    struct TargetState {
        geometry_msgs::msg::Point position;
        geometry_msgs::msg::Point velocity;
        rclcpp::Time last_update;
        int track_count = 0;
    };
    std::map<std::string, TargetState> target_states_;

    void declare_parameters()
    {
        // 弹道预测
        this->declare_parameter<double>("shooter.ballistics.projectile_speed", 15.0);
        this->declare_parameter<double>("shooter.ballistics.gravity", 9.8);

        // 置信度阈值
        this->declare_parameter<double>("shooter.min_confidence", 0.7);  
        
        // 伺服控制
        this->declare_parameter<double>("shooter.servo.max_pan_speed", 5.0);
        this->declare_parameter<double>("shooter.servo.max_tilt_speed", 3.0);
        this->declare_parameter<double>("shooter.servo.aim_threshold", 0.015);  
        
        // 击打规则
        this->declare_parameter<double>("shooter.firing.cooldown_time", 0.3);   
        
        // 动态预测
        this->declare_parameter<double>("shooter.prediction.velocity_smoothing", 0.4);
        this->declare_parameter<double>("shooter.prediction.prediction_time", 0.15);
        this->declare_parameter<bool>("shooter.prediction.enable_prediction", true);
        
        // 调试
        this->declare_parameter<bool>("shooter.debug", true);
    }

    void update_parameters()
    {
        projectile_speed_ = this->get_parameter("shooter.ballistics.projectile_speed").as_double();
        gravity_ = this->get_parameter("shooter.ballistics.gravity").as_double();

        min_confidence_ = this->get_parameter("shooter.min_confidence").as_double();
        
        max_pan_speed_ = this->get_parameter("shooter.servo.max_pan_speed").as_double();
        max_tilt_speed_ = this->get_parameter("shooter.servo.max_tilt_speed").as_double();
        aim_threshold_ = this->get_parameter("shooter.servo.aim_threshold").as_double();
        
        cooldown_duration_ = this->get_parameter("shooter.firing.cooldown_time").as_double();

        velocity_smoothing_ = this->get_parameter("shooter.prediction.velocity_smoothing").as_double();
        prediction_time_ = this->get_parameter("shooter.prediction.prediction_time").as_double();
        enable_prediction_ = this->get_parameter("shooter.prediction.enable_prediction").as_bool();
        
        debug_mode_ = this->get_parameter("shooter.debug").as_bool();
    }

    // 选择置信度最高的目标
    std::optional<vision_msgs::msg::Detection3D> select_best_target(
        const std::vector<vision_msgs::msg::Detection3D> & targets)
    {
        std::optional<vision_msgs::msg::Detection3D> best_target;
        double highest_confidence = min_confidence_;

        for (const auto & target : targets) {
            if (target.results.empty()) continue;
            
            double confidence = target.results[0].hypothesis.score;
            if (confidence > highest_confidence) {
                highest_confidence = confidence;
                best_target = target;
            }
        }
        return best_target;
    }

    // 动态预测
    geometry_msgs::msg::Point calculate_aim_point(const vision_msgs::msg::Detection3D& target) 
    {
        const auto& current_pos = target.bbox.center.position;
        std::string target_id = target.results.empty() ? "unknown" : target.results[0].hypothesis.class_id;
        
        // 更新目标状态和速度估计
        update_target_state(target_id, current_pos);
        
        auto& state = target_states_[target_id];
        
        // 根据目标跟踪质量选择预测策略
        geometry_msgs::msg::Point aim_point;
        
        if (state.track_count >= 3) {
            // 高级预测：使用速度信息
            aim_point = advanced_prediction(state);
        } else {
            // 基础预测：直接瞄准当前位置
            aim_point = basic_prediction(current_pos);
        }
        
        if (debug_mode_ && state.track_count >= 3) {
            RCLCPP_DEBUG(this->get_logger(), "目标 %s: 速度(%.2f,%.2f,%.2f) 跟踪帧:%d", 
                        target_id.c_str(), state.velocity.x, state.velocity.y, state.velocity.z, state.track_count);
        }
        
        return aim_point;
    }

    void update_target_state(const std::string& target_id, const geometry_msgs::msg::Point& current_pos)
    {
        auto current_time = this->now();
        auto& state = target_states_[target_id];
        
        if (state.track_count > 0) {
            double dt = (current_time - state.last_update).seconds();
            if (dt > 0.01 && dt < 0.5) {  // 合理的时间间隔
                // 计算瞬时速度
                geometry_msgs::msg::Point instant_velocity;
                instant_velocity.x = (current_pos.x - state.position.x) / dt;
                instant_velocity.y = (current_pos.y - state.position.y) / dt;
                instant_velocity.z = (current_pos.z - state.position.z) / dt;
                
                // 平滑滤波
                state.velocity.x = state.velocity.x * (1 - velocity_smoothing_) + instant_velocity.x * velocity_smoothing_;
                state.velocity.y = state.velocity.y * (1 - velocity_smoothing_) + instant_velocity.y * velocity_smoothing_;
                state.velocity.z = state.velocity.z * (1 - velocity_smoothing_) + instant_velocity.z * velocity_smoothing_;
            }
        }
        
        // 更新位置和时间
        state.position = current_pos;
        state.last_update = current_time;
        state.track_count++;
    }

    geometry_msgs::msg::Point basic_prediction(const geometry_msgs::msg::Point& current_pos)
    {
        // 基础预测：只考虑重力和固定提前量
        double distance = std::sqrt(current_pos.x * current_pos.x + current_pos.y * current_pos.y);
        double time_to_target = distance / projectile_speed_;
        double drop = 0.5 * gravity_ * time_to_target * time_to_target;
        
        geometry_msgs::msg::Point aim_point = current_pos;
        aim_point.z += drop;
        
        return aim_point;
    }

    geometry_msgs::msg::Point advanced_prediction(const TargetState& state)
    {
        geometry_msgs::msg::Point aim_point;

        if (enable_prediction_ && state.track_count >= 5) {

            // 高级预测：速度 + 自适应预测时间
            double distance = std::sqrt(state.position.x * state.position.x + state.position.y * state.position.y);
            double base_time = distance / projectile_speed_;
            double prediction_time = std::min(base_time, prediction_time_);
            
            // 预测位置 = 当前位置 + 速度 × 预测时间
            aim_point.x = state.position.x + state.velocity.x * prediction_time;
            aim_point.y = state.position.y + state.velocity.y * prediction_time;
            aim_point.z = state.position.z + state.velocity.z * prediction_time;
            
            // 重力补偿
            double drop = 0.5 * gravity_ * base_time * base_time;
            aim_point.z += drop;
            
            if (debug_mode_) {
                RCLCPP_DEBUG(this->get_logger(), "高级预测: 基础时间=%.3fs, 预测时间=%.3fs, 落点补偿=%.3fm", 
                            base_time, prediction_time, drop);
            }
        } else {
            // 回退到基础预测
            aim_point = basic_prediction(state.position);
        }
        
        return aim_point;
    }

    // 伺服控制
    geometry_msgs::msg::Twist compute_servo_commands(const geometry_msgs::msg::Point& aim_point)
    {
        geometry_msgs::msg::Twist cmd;

        // 计算需要转动的角度
        double desired_pan = std::atan2(aim_point.y, aim_point.x);
        double desired_tilt = std::atan2(aim_point.z, std::sqrt(aim_point.x * aim_point.x + aim_point.y * aim_point.y));

        // 计算误差
        double pan_error = desired_pan - current_pan_;
        double tilt_error = desired_tilt - current_tilt_;

        // PID控制（简化为比例控制）
        cmd.angular.z = std::clamp(pan_error * 2.5, -max_pan_speed_, max_pan_speed_);
        cmd.angular.y = std::clamp(tilt_error * 2.5, -max_tilt_speed_, max_tilt_speed_);

        // 更新当前角度
        current_pan_ += cmd.angular.z * 0.05;
        current_tilt_ += cmd.angular.y * 0.05;

        return cmd;
    }

    // 发射决策
    bool should_fire(const geometry_msgs::msg::Twist& cmd, const vision_msgs::msg::Detection3D& target)
    {
        auto now = this->now();

        // 冷却期检查
        if (now < cooldown_end_time_) {
            return false;
        }

        // 严格的瞄准精度检查
        if (std::abs(cmd.angular.z) > aim_threshold_ || std::abs(cmd.angular.y) > aim_threshold_) {
            return false;
        }

        // 目标稳定性检查
        if (target.results.empty()) return false;
        
        std::string target_id = target.results[0].hypothesis.class_id;
        if (target_states_.find(target_id) == target_states_.end()) return false;
        
        auto& state = target_states_[target_id];
        if (state.track_count < 2) {
            // 需要至少2帧跟踪以确保目标稳定
            return false;
        }

        // 速度稳定性检查 (避免在目标急转弯时射击)
        double speed = std::sqrt(state.velocity.x * state.velocity.x + state.velocity.y * state.velocity.y);
        RCLCPP_DEBUG(this->get_logger(), "速度: %.2f m/s", speed);
        if (speed > 3.0) {  // 速度过快，可能不稳定
            return false;
        }

        return true;
    }

    void setup_communication()
    {
        detection_sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/detections", 10,
            std::bind(&ShooterNode::detection_callback, this, std::placeholders::_1)
        );

        control_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        fire_pub_ = this->create_publisher<std_msgs::msg::Bool>("/fire_command", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),  
            std::bind(&ShooterNode::timer_callback, this)
        );
    }

    void detection_callback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        RCLCPP_DEBUG(this->get_logger(), "收到检测消息，包含 %zu 个目标", msg->detections.size());
        last_detection_time_ = this->now();
        
        if (msg->detections.empty()) {
            if (debug_mode_) {
                RCLCPP_DEBUG(this->get_logger(), "未检测到目标");
            }
            current_target_ = std::nullopt;
            return;
        }

        update_parameters();
        
        // 选择目标
        auto best_target = select_best_target(msg->detections);
        if (!best_target) {
            current_target_ = std::nullopt;
            return;
        }
        
        current_target_ = best_target;
        
        // 计算瞄准点
        auto aim_point = calculate_aim_point(*best_target);
        
        // 计算控制命令
        auto servo_cmd = compute_servo_commands(aim_point);
        control_pub_->publish(servo_cmd);
        
        // 发射决策
        if (should_fire(servo_cmd, *best_target)) {
            fire_shot();
        }
    }

    void fire_shot()
    {
        std_msgs::msg::Bool fire_msg;
        fire_msg.data = true;
        fire_pub_->publish(fire_msg);
        
        cooldown_end_time_ = this->now() + rclcpp::Duration::from_seconds(cooldown_duration_);
        shot_count_++;
        
        if (debug_mode_) {
            RCLCPP_INFO(this->get_logger(), "第%d次发射, 冷却时间:%.1fs", shot_count_, cooldown_duration_);
        }
    }

    void timer_callback()
    {
        auto now = this->now();

        // 清理过期的目标状态
        for (auto it = target_states_.begin(); it != target_states_.end(); ) {
            if ((now - it->second.last_update).seconds() > 2.0) {
                it = target_states_.erase(it);
            } 
            else {
                it++;
            }
        }
        
        // 状态报告
        if (debug_mode_) {
            RCLCPP_DEBUG(this->get_logger(), 
                        "云台: (%.3f, %.3f), 目标: %s, 发射: %d, 跟踪目标: %zu",
                        current_pan_, current_tilt_, 
                        current_target_ ? "有" : "无", 
                        shot_count_, target_states_.size());
        }
    }

    void initialize_state() {
        current_pan_ = 0.0;
        current_tilt_ = 0.0;
        shot_count_ = 0;
        hit_count_ = 0;
        last_detection_time_ = this->now();
        cooldown_end_time_ = this->now();
        current_target_ = std::nullopt;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ShooterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}