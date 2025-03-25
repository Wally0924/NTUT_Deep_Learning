import pandas as pd

# # 讀取 test.csv（包含圖片路徑）
# test_df = pd.read_csv("Taiwanese/tw_food_101/tw_food_101/tw_food_101_test_list.csv", header=None, names=["index", "image_path"])

# # 讀取 predict.csv（包含預測標籤）
# predict_df = pd.read_csv("/home/rvl/mingwei/NTUT_Deep_Learning/test_predictions.csv")

# # 確保 predict_df 按索引排序，防止錯配
# predict_df = predict_df.sort_values("index").reset_index(drop=True)

# # 合併 test 和 predict
# test_df["label"] = predict_df["label"]  # 將預測標籤加到 test.csv

# # 重新排列欄位順序，符合 train.csv 格式
# test_df = test_df[["index", "label", "image_path"]]

# # 儲存為新的 CSV（格式：index, label, image_path）
# test_df.to_csv("test_with_labels.csv", index=False, header=False)

# print("✅ 合併完成，已儲存為 test_with_labels.csv")

# 讀取 train.csv（包含訓練集）
train_df = pd.read_csv("Taiwanese/tw_food_101/tw_food_101/tw_food_101_train.csv", header=None, names=["index", "label", "image_path"])

# 讀取 test_with_labels.csv（包含測試集與標籤）
test_with_labels_df = pd.read_csv("test_with_labels.csv", header=None, names=["index", "label", "image_path"])

# 合併訓練集與測試集
combined_df = pd.concat([train_df, test_with_labels_df], ignore_index=True)

# 儲存為新的 CSV（合併後的 train_with_test.csv）
combined_df.to_csv("train_with_test.csv", index=False, header=False)

print("✅ 合併完成，已儲存為 train_with_test.csv")
