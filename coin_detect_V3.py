import cv2
import numpy as np

# 硬貨の特徴（色、穴の有無、サイズ範囲）
COIN_FEATURES = {
    "500": {"color": (160, 160, 160), "hole": False, "size_range": (27, 30)},  # 500円
    "100": {"color": (180, 180, 180), "hole": False, "size_range": (22, 25)},  # 100円
    "50": {"color": (180, 180, 180), "hole": True, "size_range": (20, 23)},    # 50円
    "10": {"color": (90, 60, 30), "hole": False, "size_range": (23, 26)},      # 10円
    "5": {"color": (140, 110, 50), "hole": True, "size_range": (20, 23)},      # 5円
    "1": {"color": (200, 200, 200), "hole": False, "size_range": (17, 20)},    # 1円
}

# 初期スケール（仮の値、後で校正可能）
PIXEL_TO_MM = 1.0

# マウスクリック用
clicked_points = []

def is_color_similar(color1, color2, threshold=50):
    return np.linalg.norm(np.array(color1) - np.array(color2)) < threshold

def detect_coins(frame):
    """フレームから硬貨を検出し、種類と合計金額を推定"""
    global PIXEL_TO_MM
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 円を検出
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=80
    )

    total_value = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            # 硬貨領域を切り抜き
            coin = frame[y - r:y + r, x - r:x + r]
            if coin.size == 0:
                continue

            # 平均色を計算
            avg_color = np.mean(coin, axis=(0, 1))

            # 真ん中の穴の有無を確認
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.circle(mask, (x, y), r, 255, -1)
            coin_gray = cv2.bitwise_and(gray, gray, mask=mask)
            coin_center = coin_gray[y - r // 2:y + r // 2, x - r // 2:x + r // 2]

            has_hole = np.mean(coin_center) < 50  # 中央部分が暗い場合は穴とみなす

            # サイズ（半径）をミリメートル換算
            diameter_mm = 2 * r * PIXEL_TO_MM

            # 硬貨の種類を推定
            coin_type = "undefined"
            for value, features in COIN_FEATURES.items():
                size_min, size_max = features["size_range"]
                if (
                    is_color_similar(avg_color, features["color"])
                    and has_hole == features["hole"]
                    and size_min <= diameter_mm <= size_max
                ):
                    coin_type = value
                    total_value += int(coin_type)
                    break

            # 検出結果を描画
            color = (0, 255, 0) if coin_type != "undefined" else (0, 0, 255)
            cv2.circle(frame, (x, y), r, color, 2)
            cv2.putText(
                frame,
                f"{coin_type} Yen" if coin_type != "undefined" else "undefined",
                (x - 40, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return frame, total_value

def calibrate_scale(event, x, y, flags, param):
    """マウスクリックで定規の2点を取得し、スケールを計算"""
    global clicked_points, PIXEL_TO_MM

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        if len(clicked_points) == 2:
            # 2点間のピクセル距離を計算
            pt1, pt2 = clicked_points
            pixel_distance = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

            # 実際の距離をユーザーに入力させる
            real_distance_mm = float(input("実際の定規の距離（ミリメートル）を入力してください: "))
            PIXEL_TO_MM = real_distance_mm / pixel_distance
            print(f"スケールが設定されました: 1ピクセル = {PIXEL_TO_MM:.4f} mm")
            clicked_points = []  # リセット

def main():
    global clicked_points
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Coin Detector")
    cv2.setMouseCallback("Coin Detector", calibrate_scale)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 硬貨を検出して合計金額を計算
        processed_frame, total_value = detect_coins(frame)

        # 合計金額を表示
        cv2.putText(
            processed_frame,
            f"Total: {total_value} Yen",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # 定規計測のためのクリックポイントを描画
        for point in clicked_points:
            cv2.circle(processed_frame, point, 5, (0, 0, 255), -1)

        cv2.imshow("Coin Detector", processed_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
