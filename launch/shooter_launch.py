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
        description='Enable shooter debug mode'
    )
    
    enable_firing = DeclareLaunchArgument(
        'enable_firing',
        default_value='true',
        description='Enable projectile firing'
    )
    
    # 击打节点
    shooter_node = Node(
        package='teamX_challenge',
        executable='shooter_node',
        name='shooter_node',
        output='screen',
        parameters=[
            params_file,
            {
                'shooter.debug': LaunchConfiguration('debug'),
                'shooter.firing.enabled': LaunchConfiguration('enable_firing')
            }
        ],
        remappings=[
            ('/detections', '/detections'),              # 视觉检测结果
            ('/cmd_vel', '/gimbal_cmd_vel'),             # 云台控制命令
            ('/fire_command', '/fire_command')           # 发射命令
        ]
    )
    
    return LaunchDescription([
        debug_mode,
        enable_firing,
        shooter_node
    ])