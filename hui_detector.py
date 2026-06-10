from collections import namedtuple
import math

import cv2
import numpy as np


# center_x : 컨트롤러의 조향 목표 x좌표 (BEV 기준)
# lane_x   : 검출된 차선의 원시 x좌표 (디버그용)
# angle    : 추적 차선의 기울기 (90도 = 수직/직진)
# source   : 사용한 추정 방법 ('yellow' / 'white_mid' / 'white_only' / None)
# stop     : True면 노란선이 전혀 안 보이고 흰색만 보임 → 정지해야 함
LaneInfo = namedtuple('LaneInfo', ['center_x', 'lane_x', 'angle', 'valid', 'source', 'stop'])


class LaneDetector(object):
    """트랙 구조 Y-W-Y-W-Y 에서 중앙 노란선을 추적한다.

    로직:
    1. 흰 점선 2개를 찾는다 (로봇 기준 좌/우 각 1개).
    2. 두 흰 점선 사이에 있는 노란선을 중앙 노란선으로 선택한다.
       → 바깥 노란 경계선은 흰 점선 밖에 있으므로 자동 제외된다.
    3. 중앙 노란선이 순간적으로 안 보이면 두 흰 점선의 중점을 임시 목표로 사용한다.
    4. 흰 점선도 안 보일 때만 직전 추적 위치(tracked_yellow_x) 기억으로 폴백한다.
    """

    def __init__(self):
        # 측정값: 노란선 R=140 G=140 B=70 → H≈30, S≈127, V≈140
        self.lower_yellow = np.array([15,  80,  80])
        self.upper_yellow = np.array([40, 255, 255])

        # 측정값: 흰 점선 R=G=B=160(±20) → S≈0, V≈140~180
        # ※ 여전히 안 잡히면: lower_white[2] (V) 를 100으로 낮추기
        # ※ 노이즈 너무 많으면: lower_white[2] 를 160으로 올리기
        self.lower_white = np.array([0,  0, 130])
        self.upper_white = np.array([180, 40, 255])

        self.morph_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # 흰 점선은 짧고 얇으므로 팽창 후 엣지 검출
        self.morph_kernel_sm = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 컨트롤러의 camera_offset_x와 일치시켜야 디버그 시각화가 정확합니다
        self.camera_offset_x = 0

        # 화면 하단 이 비율보다 위쪽에 있는 선분은 무시
        self.min_segment_y_ratio = 0.5
        # 같은 차선 그룹으로 묶는 x 거리 허용값(px)
        self.x_cluster_margin = 60
        # 코앞 구간만 남겨 기울기 왜곡 방지하는 세로 범위(px)
        self.front_band_height = 80

        # 흰 점선이 양쪽 모두 안 보일 때 노란선 메모리 추적용
        # (두 흰 점선이 보이는 동안에는 이 값을 사용하지 않고,
        #  공간 제약으로 노란선을 선택하므로 훨씬 신뢰도가 높다)
        self.tracked_yellow_x = None

        # 흰 점선 없이 메모리만으로 노란선 탐색할 때 허용 탐색 반경(px).
        # 이 범위를 벗어난 노란선은 바깥 경계선으로 보고 무시한다.
        # ※ 코너 이탈 후 복귀가 느리면 늘리고, 바깥선을 집으면 줄이세요.
        self.max_yellow_drift = 160

        # 노란선이 하나도 안 보일 때, 흰색이 화면의 이 비율 이상을 덮어야
        # "횡단보도/흰 바닥에 완전히 진입" 으로 판단해 정지한다.
        # 화면 거의 전체가 흰색일 때만 정지하도록 매우 타이트하게 설정.
        # ※ 그래도 코너에서 멈추면 더 올리고(예: 0.97), 정지가 안 되면 낮추세요.
        self.white_stop_ratio = 0.92

        # 바깥 외곽 노란선이 한쪽만 보일 때, 흰 점선 위치에서 안쪽으로
        # 이만큼(px) 떨어진 지점을 강제 복귀 목표로 삼는다.
        # ※ 복귀가 약하면(여전히 바깥선 쪽으로 붙으면) 값을 키우세요.
        self.recovery_margin = 60

    # ---------------- 전처리 ----------------
    def bev_transform(self, src):
        height, width = src.shape[:2]
        tl = (int(width * 0.31), int(height * 0.58))
        tr = (int(width * 0.69), int(height * 0.58))
        br = (int(width * 0.95), int(height * 0.95))
        bl = (int(width * 0.05), int(height * 0.95))

        src_points = np.float32([tl, tr, br, bl])

        bev_width = int(width * 0.5)
        bev_height = height
        bev_x_min = int((width - bev_width) / 2)
        bev_x_max = bev_x_min + bev_width

        dst_points = np.float32([
            (bev_x_min, 0), (bev_x_max, 0),
            (bev_x_max, bev_height), (bev_x_min, bev_height),
        ])

        m = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(src, m, (width, height))

    def extract_yellow_mask(self, bev):
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

    def extract_white_mask(self, bev):
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.lower_white, self.upper_white)

    def detect_edges(self, mask):
        """노란선(굵고 긴 실선)용 엣지 검출."""
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        blur = cv2.GaussianBlur(closed, (5, 5), 0)
        return cv2.Canny(blur, 50, 150)

    def detect_edges_white(self, mask):
        """흰 점선(짧고 얇은 선)용 엣지 검출. 먼저 팽창해 얇은 선을 두껍게 만든다."""
        dilated = cv2.dilate(mask, self.morph_kernel_sm, iterations=1)
        closed  = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, self.morph_kernel_sm)
        blur    = cv2.GaussianBlur(closed, (5, 5), 0)
        return cv2.Canny(blur, 30, 100)  # 임계값도 낮춰 희미한 선 검출

    def detect_hough_lines_p(self, edges):
        """노란선(실선)용: 엄격한 파라미터."""
        return cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=30,
                               minLineLength=20, maxLineGap=40)

    def detect_hough_lines_p_white(self, edges):
        """흰 점선용: 짧은 선분도 검출하도록 완화된 파라미터.
        threshold 낮춤(15), minLineLength 단축(10), maxLineGap 확대(60).
        """
        return cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=15,
                               minLineLength=10, maxLineGap=60)

    # ---------------- 마스크 → 선분 그룹 목록 ----------------
    def _detect_groups(self, mask, img_height, is_white=False):
        """마스크에서 엣지→허프→x좌표 클러스터링까지 처리해
        [(mean_x, mean_angle, [lines]), ...] 목록을 x 오름차순으로 반환한다.
        is_white=True 이면 점선에 최적화된 엣지/허프 파라미터를 사용한다."""
        if is_white:
            edges = self.detect_edges_white(mask)
            lines = self.detect_hough_lines_p_white(edges)
        else:
            edges = self.detect_edges(mask)
            lines = self.detect_hough_lines_p(edges)
        if lines is None:
            return []

        candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            bx = x1 if y1 > y2 else x2
            by = max(y1, y2)
            if by < img_height * self.min_segment_y_ratio:
                continue
            candidates.append((bx, by, line))

        if not candidates:
            return []

        candidates.sort(key=lambda c: c[0])

        clusters = [[candidates[0]]]
        for c in candidates[1:]:
            if c[0] - clusters[-1][-1][0] <= self.x_cluster_margin:
                clusters[-1].append(c)
            else:
                clusters.append([c])

        groups = []
        for cluster in clusters:
            max_by = max(c[1] for c in cluster)
            front = [c for c in cluster if c[1] >= max_by - self.front_band_height]

            xs = [c[0] for c in front]
            angles = []
            for bx, by, line in front:
                x1, y1, x2, y2 = line[0]
                if y1 < y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                angles.append(math.degrees(math.atan2(y1 - y2, x2 - x1)))

            groups.append((float(np.mean(xs)), float(np.mean(angles)),
                           [c[2] for c in cluster]))

        groups.sort(key=lambda g: g[0])
        return groups

    # ---------------- 메인 처리 ----------------
    def process(self, src):
        height, width = src.shape[:2]
        bev = self.bev_transform(src)

        yellow_mask = self.extract_yellow_mask(bev)
        white_mask  = self.extract_white_mask(bev)

        robot_cx = width / 2.0 - self.camera_offset_x

        # ── 1단계: 흰 점선 2개 찾기 (로봇 좌/우 가장 가까운 것) ─────────
        white_groups = self._detect_groups(white_mask, height, is_white=True)

        # 로봇 왼쪽 흰선 중 가장 오른쪽 것 (로봇과 가장 가까운 왼쪽 선)
        left_white = max(
            (g for g in white_groups if g[0] < robot_cx),
            key=lambda g: g[0], default=None)
        # 로봇 오른쪽 흰선 중 가장 왼쪽 것 (로봇과 가장 가까운 오른쪽 선)
        right_white = min(
            (g for g in white_groups if g[0] >= robot_cx),
            key=lambda g: g[0], default=None)

        # ── 2단계: 바깥 외곽 노란선 식별 ─────────────────────────────────
        # 트랙 구조: [바깥노랑][도로][흰점선][도로][중앙노랑][도로][흰점선][도로][바깥노랑]
        # → 바깥 노란선은 항상 흰 점선보다 더 바깥쪽(로봇에서 먼 쪽)에 있다.
        #   따라서 흰 점선 기준 바깥쪽에 있는 노란선은 절대 중앙선 후보가 될 수 없다.
        yellow_groups = self._detect_groups(yellow_mask, height)

        left_outer = None
        right_outer = None
        if left_white:
            left_outer = max(
                (g for g in yellow_groups if g[0] < left_white[0]),
                key=lambda g: g[0], default=None)
        if right_white:
            right_outer = min(
                (g for g in yellow_groups if g[0] > right_white[0]),
                key=lambda g: g[0], default=None)

        # 중앙선 후보에서 바깥 외곽선은 항상 제외
        inner_yellow_groups = [g for g in yellow_groups
                                if g is not left_outer and g is not right_outer]

        # ── 3단계: 두 흰 점선 사이의 노란선 = 중앙 노란선 ───────────────
        center_yellow = None

        if left_white and right_white:
            # 흰 점선 양쪽 모두 보임: 공간 제약으로 중앙 노란선 선택
            between = [g for g in inner_yellow_groups
                       if left_white[0] < g[0] < right_white[0]]
            if between:
                white_mid = (left_white[0] + right_white[0]) / 2.0
                center_yellow = min(between, key=lambda g: abs(g[0] - white_mid))
                self.tracked_yellow_x = center_yellow[0]  # 메모리 갱신
            # between이 비어 있으면: 중앙 노란선 순간 미검출 → 4단계에서 white_mid 사용

        elif left_white and not right_white:
            # 왼쪽 흰 점선만 보임: 중앙 노란선은 그 오른쪽에 있어야 함
            ref = self.tracked_yellow_x if self.tracked_yellow_x is not None else robot_cx
            candidates = [g for g in inner_yellow_groups if g[0] > left_white[0]]
            if candidates:
                center_yellow = min(candidates, key=lambda g: abs(g[0] - ref))
                self.tracked_yellow_x = center_yellow[0]

        elif right_white and not left_white:
            # 오른쪽 흰 점선만 보임: 중앙 노란선은 그 왼쪽에 있어야 함
            ref = self.tracked_yellow_x if self.tracked_yellow_x is not None else robot_cx
            candidates = [g for g in inner_yellow_groups if g[0] < right_white[0]]
            if candidates:
                center_yellow = min(candidates, key=lambda g: abs(g[0] - ref))
                self.tracked_yellow_x = center_yellow[0]

        else:
            # 흰 점선 없음: 마지막으로 추적한 위치에서 max_yellow_drift 이내만 허용
            ref = self.tracked_yellow_x if self.tracked_yellow_x is not None else robot_cx
            close_groups = [g for g in inner_yellow_groups
                            if abs(g[0] - ref) < self.max_yellow_drift]
            if close_groups:
                center_yellow = min(close_groups, key=lambda g: abs(g[0] - ref))
                self.tracked_yellow_x = center_yellow[0]

        # ── 4단계: 조향 목표 결정 ────────────────────────────────────────
        # 안전장치 우선: 바깥 외곽선이 한쪽만 보이면 그쪽으로 너무 붙은 것
        # → 중앙 노란선 추적 결과와 무관하게 무조건 반대쪽(안쪽)으로 복귀
        if left_outer and not right_outer:
            ref_x    = left_white[0] if left_white else left_outer[0]
            center_x = ref_x + self.recovery_margin
            lane_x   = left_outer[0]
            angle    = left_outer[1]
            source   = 'avoid_left_outer'
            valid    = True
            self.tracked_yellow_x = center_x

        elif right_outer and not left_outer:
            ref_x    = right_white[0] if right_white else right_outer[0]
            center_x = ref_x - self.recovery_margin
            lane_x   = right_outer[0]
            angle    = right_outer[1]
            source   = 'avoid_right_outer'
            valid    = True
            self.tracked_yellow_x = center_x

        elif center_yellow:
            # [주] 중앙 노란선 추적
            center_x = center_yellow[0]
            lane_x   = center_x
            angle    = center_yellow[1]
            source   = 'yellow'
            valid    = True

        elif left_white and right_white:
            # [임시] 중앙 노란선 순간 미검출 → 흰 점선 중점을 임시 목표로
            center_x = (left_white[0] + right_white[0]) / 2.0
            lane_x   = center_x
            angle    = (left_white[1] + right_white[1]) / 2.0
            source   = 'white_mid'
            valid    = True

        else:
            center_x = lane_x = angle = source = None
            valid    = False

        # ── 정지 조건: 노란선이 전혀 없고, 흰색이 화면의 상당 부분을 덮음 ──
        # (코너 회전 중 잠깐 흰 점선만 보이는 정도로는 멈추지 않도록
        #  흰색 비율이 충분히 클 때만 "흰 바닥"으로 간주)
        white_ratio = float(np.count_nonzero(white_mask)) / white_mask.size
        stop = (not yellow_groups) and (white_ratio > self.white_stop_ratio)

        info = LaneInfo(
            center_x=center_x, lane_x=lane_x,
            angle=angle, valid=valid, source=source, stop=stop,
        )

        debug_images = {
            'Lane BEV':         bev,
            'Lane Yellow Mask': yellow_mask,
            'Lane White Mask':  white_mask,
            'Lane Detection (BEV)': self._draw_debug(
                bev, info, yellow_groups, left_white, right_white, center_yellow, width),
        }
        return info, debug_images

    def _draw_debug(self, bev, info, yellow_groups,
                    left_white, right_white, center_yellow, img_width):
        dst = bev.copy()
        height, width = dst.shape[:2]
        robot_cx = int(width / 2.0 - self.camera_offset_x)

        # 모든 노란선 그룹 (얇게, 어두운 노랑)
        for g in yellow_groups:
            for line in g[2]:
                x1, y1, x2, y2 = line[0]
                cv2.line(dst, (x1, y1), (x2, y2), (0, 180, 180), 1)

        # 선택된 중앙 노란선 (굵게, 밝은 노랑)
        if center_yellow:
            for line in center_yellow[2]:
                x1, y1, x2, y2 = line[0]
                cv2.line(dst, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 좌/우 흰 점선
        if left_white:
            for line in left_white[2]:
                x1, y1, x2, y2 = line[0]
                cv2.line(dst, (x1, y1), (x2, y2), (200, 200, 200), 2)
        if right_white:
            for line in right_white[2]:
                x1, y1, x2, y2 = line[0]
                cv2.line(dst, (x1, y1), (x2, y2), (200, 200, 200), 2)

        # 두 흰 점선 사이 유효 구간 표시
        if left_white and right_white:
            lx, rx = int(left_white[0]), int(right_white[0])
            overlay = dst.copy()
            cv2.rectangle(overlay, (lx, 0), (rx, height), (40, 40, 80), -1)
            cv2.addWeighted(overlay, 0.25, dst, 0.75, 0, dst)
            cv2.line(dst, (lx, 0), (lx, height), (180, 180, 255), 1)
            cv2.line(dst, (rx, 0), (rx, height), (180, 180, 255), 1)

        # 파란 세로선: 이미지 정중앙 / 초록 세로선: 로봇 위치
        cv2.line(dst, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)
        cv2.line(dst, (robot_cx, 0), (robot_cx, height), (0, 200, 0), 1)

        if info.valid:
            y_mark = height - 20
            if left_white:
                cv2.circle(dst, (int(left_white[0]),  y_mark), 7, (220, 220, 220), -1)
            if right_white:
                cv2.circle(dst, (int(right_white[0]), y_mark), 7, (220, 220, 220), -1)
            if center_yellow:
                cv2.circle(dst, (int(center_yellow[0]), y_mark), 7, (0, 230, 230), -1)
            # 빨간 원: 최종 조향 목표
            cv2.circle(dst, (int(info.center_x), y_mark), 9, (0, 0, 255), -1)
            # 화살표: 로봇 → 목표
            cv2.arrowedLine(dst, (robot_cx, y_mark), (int(info.center_x), y_mark),
                            (0, 255, 255), 2, tipLength=0.3)

            src_color = {'yellow': (0, 255, 255), 'white_mid': (200, 200, 200)}
            color = src_color.get(info.source, (255, 255, 255))
            label = (f'[{info.source}] ang={info.angle:.1f} '
                     f'cx={info.center_x:.0f} err={robot_cx - info.center_x:.0f}')
            cv2.putText(dst, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        status = []
        if left_white:  status.append('L-white')
        if right_white: status.append('R-white')
        if center_yellow: status.append('C-yellow')
        cv2.putText(dst, ' '.join(status) if status else 'NO DETECT',
                    (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (180, 255, 180) if status else (0, 0, 255), 1)

        return dst
