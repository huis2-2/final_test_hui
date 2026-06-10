import cv2
import numpy as np


class TrafficLightDetector:
    """신호등 ROI 영역을 잘라내어 빨강/초록 신호를 색상 기반으로 판정한다."""

    RED = 'RED'
    GREEN = 'GREEN'
    UNKNOWN = 'UNKNOWN'

    def __init__(self):
        # 신호등이 비치는 영역 비율 (x1, y1, x2, y2) - 카메라 위치에 맞게 조정
        self.roi_ratio = (0.35, 0.0, 0.65, 0.35)

        # 빨강은 H값이 0 부근과 180 부근 양쪽에 걸쳐 있어 두 구간으로 나눠서 검출
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

        self.lower_green = np.array([45, 80, 80])
        self.upper_green = np.array([85, 255, 255])

        # ROI 내에서 해당 색상으로 판정하기 위한 최소 픽셀 비율
        self.min_pixel_ratio = 0.02

    def get_roi_box(self, width, height):
        x1r, y1r, x2r, y2r = self.roi_ratio
        return (int(width * x1r), int(height * y1r),
                int(width * x2r), int(height * y2r))

    def detect(self, src):
        """src(BGR) 전체 이미지를 받아 (signal, debug_info)를 반환한다."""
        height, width = src.shape[:2]
        x1, y1, x2, y2 = self.get_roi_box(width, height)
        roi = src[y1:y2, x1:x2]
        if roi.size == 0:
            return self.UNKNOWN, None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.lower_red1, self.upper_red1),
            cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        )
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        roi_area = roi.shape[0] * roi.shape[1]
        red_ratio = cv2.countNonZero(red_mask) / roi_area
        green_ratio = cv2.countNonZero(green_mask) / roi_area

        if red_ratio < self.min_pixel_ratio and green_ratio < self.min_pixel_ratio:
            signal = self.UNKNOWN
        elif red_ratio >= green_ratio:
            signal = self.RED
        else:
            signal = self.GREEN

        debug = {
            'roi_box': (x1, y1, x2, y2),
            'red_ratio': red_ratio,
            'green_ratio': green_ratio,
        }
        return signal, debug

    def draw_debug(self, src, signal, debug):
        """ROI 박스와 판정 결과를 원본 이미지 위에 그려서 반환한다."""
        dst = src.copy()
        if debug is None:
            return dst

        x1, y1, x2, y2 = debug['roi_box']
        if signal == self.RED:
            color = (0, 0, 255)
        elif signal == self.GREEN:
            color = (0, 255, 0)
        else:
            color = (200, 200, 200)

        cv2.rectangle(dst, (x1, y1), (x2, y2), color, 2)
        label = f'{signal} (R:{debug["red_ratio"]:.2f} G:{debug["green_ratio"]:.2f})'
        cv2.putText(dst, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return dst
