# 圆环检测视觉节点 (Ring Vision Node)

## 技术报告语雀文档
https://www.yuque.com/ruizhidehoutou/lcigge/epy45iukmd9tg58y?singleDoc# 《技术报告》

## 概述

这是一个基于ROS 2的圆环检测视觉节点，使用OpenCV的霍夫圆检测算法实时检测图像中的圆环目标，并将检测结果发布为多目标消息。

## 依赖项

### ROS 2 包
- `rclcpp`
- `sensor_msgs` 
- `cv_bridge` 
- `geometry_msgs` 

### 自定义消息
- `referee_pkg/msg/MultiObject`
- `referee_pkg/msg/Object`

### 系统依赖
- OpenCV 4.x
- C++17 或更高版本

## 编译指令
```bash
colcon build --packages-select my_package
source install/setup.bash
```

## 运行指令
每次打开新的终端时，都要运行以下指令更新一下环境
```bash
source install/setup.bash
```

### 运行摄像头仿真
在第一个终端中运行
```bash
ros2 launch camera_sim_pkg camera.launch.py
```

### 运行vision_node（含参数调试）
在第二个终端运行
```bash
ros2 run my_package vision_node --ros-args -p vision.min_radius:=15 -p vision.max_radius:=80 -p 
```
或
```bash
ros2 launch my_package vision.launch.py
```

### 运行裁判系统
在第三个终端运行
```bash
ros2 launch referee_pkg referee_pkg_launch.xml TeamName:="TEAM9"
```

## 算法原理

### 图像处理流程

1. 图像获取：订阅 /camera/image_raw 话题获取原始图像
2. 预处理：
   · 转换为灰度图像
   · 应用高斯模糊降噪
3. 圆环检测：
   · 使用霍夫梯度法检测圆环
   · 根据参数过滤半径范围内的圆
4. 结果处理：
   · 按半径大小排序检测到的圆环
   · 选择最大的两个圆作为外圆和内圆
5. 消息发布：将检测结果发布到 /vision/target 话题

### 关键参数

· vision.min_radius：检测圆环的最小半径（默认：20像素）
· vision.max_radius：检测圆环的最大半径（默认：100像素）

### 检测逻辑

· 检测到2个或更多圆环时，选择半径最大的两个分别作为外圆和内圆
· 只检测到1个圆环时，将该圆环同时作为外圆和内圆发布
· 未检测到圆环时，发布空消息

