import rclpy
from rclpy.node import Node
from crazyflie_driver.msg import FullState
from geometry_msgs.msg import Twist

class CrazyflieMockNode(Node):
    def __init__(self):
        super().__init__('crazyflie_mock_node')

        # Publisher for telemetry data
        self.telemetry_pub = self.create_publisher(FullState, '/crazyflie/telemetry', 10)

        # Timer to publish fake telemetry data
        self.timer = self.create_timer(1.0, self.publish_telemetry)

        # Initialize the mock state values
        self.x = 0.0
        self.y = 0.0
        self.z = 1.0  # Starting at 1 meter altitude
        self.yaw = 0.0
        self.vx = 0.0  # No velocity in the x direction
        self.vy = 0.0  # No velocity in the y direction
        self.vz = 0.0  # No velocity in the z direction
        self.vyaw = 0.0  # No yaw velocity

    def publish_telemetry(self):
        # Create and fill the FullState message with mock data
        msg = FullState()
        msg.x = self.x
        msg.y = self.y
        msg.z = self.z
        msg.yaw = self.yaw
        msg.vx = self.vx
        msg.vy = self.vy
        msg.vz = self.vz
        msg.vyaw = self.vyaw

        # Log to confirm the data being published
        self.get_logger().info(f"Publishing telemetry: x={self.x}, y={self.y}, z={self.z}, yaw={self.yaw}")

        # Publish the message
        self.telemetry_pub.publish(msg)

        # Simulate some movement
        self.x += 0.1
        self.y += 0.1
        self.z = 1.0  # Keep it at a fixed altitude
        self.yaw += 0.05

def main(args=None):
    rclpy.init(args=args)
    node = CrazyflieMockNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
