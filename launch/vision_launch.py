from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包路径
    pkg_path = get_package_share_directory('teamX_challenge')
    
    # 参数文件路径
    params_file = os.path.join(pkg_path, 'config', 'params.yaml')
    
    # 声明启动参数
    debug_mode = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug mode'
    )
    
    show_windows = DeclareLaunchArgument(
        'show_windows', 
        default_value='true',
        description='Show visualization windows'
    )
    
    # 视觉节点
    vision_node = Node(
        package='teamX_challenge',
        executable='vision_node',
        name='competition_vision_node',
        output='screen',
        parameters=[
            params_file,
            {
                'debug.show_windows': LaunchConfiguration('show_windows'),
                'debug.log_detections': LaunchConfiguration('debug')
            }
        ],
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),  # 图像话题
            ('/detections', '/detections')               # 检测结果话题
        ]
    )
    
    return LaunchDescription([
        debug_mode,
        show_windows,
        vision_node
    ])