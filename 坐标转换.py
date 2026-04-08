import json
import copy
import itertools
import numpy as np
import pandas as pd

# ==============================================
# 【你原版的归一化函数】完全一致
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
# 批量处理JSON → 输出CSV
# ==============================================
def json_hand_landmarks_to_csv(json_path, output_csv="hand_dataset.csv"):
    # 1. 读取JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []

    # 2. 遍历每一条数据
    for key, item in data.items():
        hand_landmarks_list = item.get("hand_landmarks", [])  # 可能是1只手 or 2只手
        labels = item.get("labels", ["unknown"])
        label = labels[0]  # 取第一个标签（call、1、2、rock...）

        # 遍历每一只手（支持单手+双手）
        for hand_idx, landmarks_21 in enumerate(hand_landmarks_list):
            # 检查必须是21点
            if len(landmarks_21) != 21:
                continue

            # 3. 【核心】归一化处理 → 42维向量
            norm_42 = pre_process_landmark(landmarks_21)

            # 4. 拼装一行：标签 + 42个特征
            row = [label] + norm_42
            rows.append(row)

    # 5. 构建CSV列名
    columns = ["label"] + [f"x{i//2}_{i%2}" for i in range(42)]

    # 6. 保存CSV
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✅ 处理完成！共 {len(rows)} 条手势数据")
    print(f"✅ 已保存到：{output_csv}")
    print(f"✅ 格式：label + 42个归一化特征（手腕为原点）")

# ==============================================
# 【使用方法】
# 把 your_data.json 换成你的JSON路径
# ==============================================
if __name__ == "__main__":
    json_hand_landmarks_to_csv(
        json_path="/home/spring/hand/data/json/train/peace_inverted.json",  # 你的JSON文件
        output_csv="/home/spring/hand/data/21point/peace_inverted.csv"  # 输出CSV
    )