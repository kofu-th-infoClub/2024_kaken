import cv2
import numpy as np

# 硬貨の色と穴の有無による特徴（調整可能）
COIN_FEATURES = {
    "500": {"color": (13, 1, 2), "hole": False},  # 500円: 銀色, 穴なし
    "100": {"color": (140, 140, 140), "hole": False},  # 100円: 銀色, 穴なし
    "50": {"color": (180, 180, 180), "hole": True},    # 50円: 銀色, 穴あり
    "10": {"color": (90, 60, 30), "hole": False},      # 10円: 銅色, 穴なし
    "5": {"color": (140, 110, 50), "hole": True},      # 5円: 金色, 穴あり
    "1": {"color": (100, 100,100), "hole": False},    # 1円: 明るい銀色, 穴なし
}

# 色の近さを計算する関数
def is_color_similar(color1, color2, threshold=50):
    return np.linalg.norm(np.array(color1) - np.array(color2)) < threshold

def detect_coins(frame):
    """フレームから硬貨を検出し、種類と合計金額を推定"""
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

            # 硬貨の種類を推定
            coin_type = "undefined"
            for value, features in COIN_FEATURES.items():
                if is_color_similar(avg_color, features["color"]) and has_hole == features["hole"]:
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

def main():
    cap = cv2.VideoCapture(0)

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

        cv2.imshow("Coin Detector", processed_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
