import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# MediaPipe 手部骨架连线
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
    (5, 9), (9, 13), (13, 17)  # 手掌
]

def draw_hand(keypoints, true_label, pred_label, confidence, idx):
    plt.cla()
    plt.gca().invert_yaxis()  # y轴向下
    
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    plt.scatter(x, y, c='red', s=40)
    
    # 画点序号
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i), fontsize=8)
    
    # 画连线
    for s, e in CONNECTIONS:
        plt.plot([x[s], x[e]], [y[s], y[e]], 'b-')
    
    plt.title(f"Row {idx} | TRUE: {true_label} | PRED: {pred_label} | CONF: {confidence:.2f}")
    plt.axis('equal')
    plt.pause(0.01)

def browse_inference_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(f"总行数: {len(df)}")

    plt.ion()
    plt.figure(figsize=(6, 6))

    for idx, row in df.iterrows():
        # 你的格式：第4列开始是 p0~p41
        test_id=['test_id']
        original_row_id=['original_row_id']
        true_label = row['true_label']
        pred_label = row['pred_label']
        confidence = row['confidence']
        
        # 取出 p0 ~ p41 共42个值
        coords = row[5:47].values.astype(float)
        kp = coords.reshape(21, 2)

        draw_hand(kp, true_label, pred_label, confidence, idx)

        cmd = input("回车下一张，q退出：").strip().lower()
        if cmd == 'q':
            break

    plt.close()
    plt.ioff()

# ========== 在这里改你的CSV路径 ==========
if __name__ == "__main__":
    browse_inference_csv("/home/spring/hand/hand-gesture-recognition-using-mediapipe/run/middle_finger_no_gesture_confusion.csv")