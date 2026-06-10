import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from final_test.hui_signal import TrafficLightDetector
from final_test.hui_detector import LaneDetector
from final_test.hui_controller import LaneKeepingController


class HuiMainNode(Node):
    def __init__(self):
        super().__init__('hui_main_node')

        self.bridge = CvBridge()

        # 실차(yahboom) 기준 기본 토픽: USB 카메라는 /usb_cam/image_raw, 구동은 /cmd_vel
        # (시뮬레이터로 되돌릴 때는 launch/파라미터로 furosim 토픽을 넘기면 된다)
        self.declare_parameter('camera_topic', '/usb_cam/image_raw')
        self.camera_topic = self.get_parameter('camera_topic').value

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        # 신호등이 아직 설치되지 않아 값을 받을 수 없는 동안에는 False로 두면
        # 신호등 판정을 건너뛰고 차선 주행만 수행한다. 설치 후 True로 바꾸면 된다.
        self.declare_parameter('use_traffic_light', False)
        self.use_traffic_light = self.get_parameter('use_traffic_light').value

        self.sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.signal_detector = TrafficLightDetector()
        self.lane_detector = LaneDetector()
        self.controller = LaneKeepingController()

        self.get_logger().info('hui_main_node started')

    def image_callback(self, msg: Image):
        try:
            src = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge 변환 실패: {e}')
            return

        # 카메라가 오른쪽에 치우쳐 달린 경우 왼쪽 여백을 잘라내고
        # 높이 기준 정사각형(오른쪽 정렬)으로 만들어 BEV 왜곡을 줄인다.
        h, w = src.shape[:2]
        if w > h:
            src = src[:, w - h:]

        width = src.shape[1]

        # 1. 신호등 색상 판정 (ROI) - 신호등 미설치 시에는 건너뛰고 차선 주행만 수행
        if self.use_traffic_light:
            signal, signal_debug = self.signal_detector.detect(src)
        else:
            signal, signal_debug = TrafficLightDetector.UNKNOWN, None

        # 2. 신호에 따라 정지 / 차선 주행 분기
        lane_debug = None
        if signal == TrafficLightDetector.RED:
            twist_msg = Twist()  # 빨간불: 정지 (linear.x = angular.z = 0)
            self.get_logger().info('RED signal -> STOP', throttle_duration_sec=1.0)
        else:
            lane_info, lane_debug = self.lane_detector.process(src)
            twist_msg = self.controller.compute_command(lane_info, width)

        # 3. 제어 명령 발행
        self.cmd_pub.publish(twist_msg)

        # 4. 디버깅 시각화
        self._show_debug(src, signal, signal_debug, lane_debug)

    def _show_debug(self, src, signal, signal_debug, lane_debug):
        signal_view = self.signal_detector.draw_debug(src, signal, signal_debug)
        cv2.imshow('Signal ROI', signal_view)

        if lane_debug is not None:
            # 차선 검출의 각 처리 단계(BEV, 색상 마스크, 모폴로지, 엣지 등)를
            # 단계별로 별도의 ROI 창에 띄워 확인할 수 있게 한다.
            for window_name, image in lane_debug.items():
                cv2.imshow(window_name, image)

        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = HuiMainNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
