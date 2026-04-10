import json
import copy
import itertools
import numpy as np
import pandas as pd

# ==============================================
# 【归一化函数】不变
# ==============================================
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 相对坐标（手腕为0,0）
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 展平一维
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # 归一化到 [-1,1]
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n): return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# ==============================================
# 【已修改】支持：一张图两只手 + 两个标签
# ==============================================
def json_hand_landmarks_to_csv(json_path, output_csv="hand_dataset.csv"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []

    for key, item in data.items():
        hand_landmarks_list = item.get("hand_landmarks", [])  # [手1, 手2]
        labels = item.get("labels", [])                       # [标签1, 标签2]

        # 🔥 核心：第几只手 → 对应第几个标签
        for hand_idx, landmarks_21 in enumerate(hand_landmarks_list):
            if len(landmarks_21) != 21:
                continue

            # 🔥 关键修改：手0 → 标签0，手1 → 标签1
            if hand_idx < len(labels):
                label = labels[hand_idx]
            else:
                label = "unknown"  # 标签不够用时兜底

            # 归一化
            norm_42 = pre_process_landmark(landmarks_21)

            # 拼接一行
            row = [label] + norm_42
            rows.append(row)

    # 列名
    columns = ["label"] + [f"p{i}" for i in range(42)]

    # 保存CSV
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✅ 处理完成！共 {len(rows)} 条手势数据")
    print(f"✅ 已保存到：{output_csv}")
    print(f"✅ 支持：一张图两只手，各自对应自己的标签！")

# ==============================================
# 使用方法
# ==============================================
if __name__ == "__main__":
    json_hand_landmarks_to_csv(
        json_path="/home/spring/hand/data/json/train/peace_inverted.json",
        output_csv="/home/spring/hand/data/21point/peace_inverted.csv"
    )