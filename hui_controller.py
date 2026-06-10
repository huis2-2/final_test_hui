from geometry_msgs.msg import Twist


class LaneKeepingController(object):
    """추정된 차로 중심 위치(center_x)와 차선 기울기(angle)를 바탕으로
    조향/속도 명령을 생성한다.

    - 코너 진입(기울기 오차가 큼) 시 더 높은 게인을 써서 적극적으로 꺾는다.
    - 각속도가 클수록 속도를 줄여 코너 안정성을 높인다.
    - 차선을 잠시 놓치면 조향각을 0으로 두고 똑바로 직진하여 통과한다.

    ※ 게인 튜닝 가이드
      - 코너에서 덜 꺾인다 → Kp_lateral_corner / Kp_heading_corner 올리기
      - 직선에서 흔들린다 → Kp_lateral / Kp_heading 낮추기
      - 속도가 너무 빠르다 → base_speed 낮추기 / speed_drop_factor 올리기
    """

    def __init__(self):
        # 인식(BEV)은 정확한데 실차가 정면 기준 한쪽으로 쏠려 달리는 경우,
        # 로봇의 실제 중심선과 이미지 중심이 어긋난 것 → 여기서 픽셀 단위로 보정.
        # ※ 이 값을 너무 크게 잡으면(예: 50px) lateral_error가 과도하게 커져
        #   곡선 구간에서 heading_error 보정 방향까지 뒤집어버릴 수 있다.
        #   작은 값(0~15px)으로만 사용하고, 직선 드리프트는 아래
        #   steering_trim(각속도 상시 보정)으로 따로 처리한다.
        self.camera_offset_x = 15

        # 직선에서 차가 한쪽으로 쏠리는 기계적 편향 보정용 상시 각속도 트림.
        # 차가 "왼쪽"으로 쏠리면 이 값을 양수 방향으로(+0.02씩) 늘려가며 테스트.
        # (코너 보정량(0.3~0.6)에 비해 작게 유지해야 곡선 판단이 안 뒤집힌다)
        self.steering_trim = 0.0

        # ── 직선 구간 게인 ────────────────────────────────────────────────
        self.Kp_lateral  = 0.005
        self.Kp_heading  = 0.022

        # ── 데드밴드: 작은 오차는 무시해 미세 떨림(와리가리) 억제 ──────────
        self.lateral_error_deadband = 8.0    # px
        self.heading_error_deadband = 1.5    # deg

        # ── 코너 구간 게인 (heading_error > threshold 일 때 전환) ─────────
        self.Kp_lateral_corner  = 0.012
        self.Kp_heading_corner  = 0.050
        self.corner_heading_threshold = 5.0   # 이 각도(°) 이상이면 코너 게인 사용

        # ── 속도 설정 ─────────────────────────────────────────────────────
        self.max_angular_speed  = 1.5
        self.base_speed         = 0.28       # 기본 전진 속도 (m/s)
        self.min_speed          = 0.14        # 최소 속도 (급코너에서 이 속도로 유지)
        # angular_z 가 클수록 속도를 이만큼 더 줄인다
        # base_speed - angular_z * factor → factor가 클수록 코너에서 더 느려짐
        self.speed_drop_factor  = 0.40

        # 차선을 잠시 놓쳤을 때 사용할 직전 조향/속도 명령 (초기값: 직진)
        self.last_angular_z = 0.0
        self.last_linear_x  = self.base_speed

    def compute_command(self, lane_info, img_width):
        twist_msg = Twist()

        if lane_info.stop:
            # 흰색이 화면 대부분을 덮음(횡단보도 등) → 정지
            twist_msg.linear.x  = 0.0
            twist_msg.angular.z = 0.0
            self.last_angular_z = 0.0
            self.last_linear_x  = 0.0
            return twist_msg

        if not lane_info.valid or lane_info.center_x is None:
            # 차선을 잠시 놓치면 직전 조향/속도를 그대로 유지
            # (애매한 흰색 구간 등에서 갑자기 직진/정지로 튀는 것 방지)
            twist_msg.linear.x  = self.last_linear_x
            twist_msg.angular.z = self.last_angular_z
            return twist_msg

        robot_center_x = (img_width / 2.0) - self.camera_offset_x
        lateral_error  = robot_center_x - lane_info.center_x
        heading_error  = lane_info.angle - 90.0

        # 작은 오차는 0으로 무시 (미세 떨림 방지)
        if abs(lateral_error) < self.lateral_error_deadband:
            lateral_error = 0.0
        if abs(heading_error) < self.heading_error_deadband:
            heading_error = 0.0

        if abs(heading_error) > self.corner_heading_threshold:
            kp_lat, kp_hdg = self.Kp_lateral_corner, self.Kp_heading_corner
        else:
            kp_lat, kp_hdg = self.Kp_lateral, self.Kp_heading

        angular_z = kp_lat * lateral_error + kp_hdg * heading_error + self.steering_trim
        angular_z = max(min(angular_z, self.max_angular_speed), -self.max_angular_speed)

        linear_x = max(self.min_speed,
                       self.base_speed - abs(angular_z) * self.speed_drop_factor)

        twist_msg.linear.x  = float(linear_x)
        twist_msg.angular.z = float(angular_z)

        self.last_angular_z = angular_z
        self.last_linear_x  = linear_x
        return twist_msg
