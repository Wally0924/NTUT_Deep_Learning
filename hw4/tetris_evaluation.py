import numpy as np
import socket
import cv2
import subprocess
import gymnasium as gym
from gymnasium import spaces
import torch
import os
import shutil
import glob
import imageio
import time
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv # 也可能需要 DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from hw4_tetris_ppo import TetrisEnv

# --- 評估腳本的配置 ---
MODEL_PATH = "/home/rvl/mingwei/NTUT_Deep_Learning/hw4/tetris_ppo_model_focused_train_final.zip"  # 指向您保存的最佳模型

N_EVAL_ENVS = 4  # 評估時使用的並行環境數量，可以設置為1以方便調試或觀察單局遊戲
N_EVAL_EPISODES = 10 # 評估運行的總回合數
TEST_STEPS_PER_EPISODE_LIMIT = 5000 # 可選：為每個評估回合設置一個最大步數限制，防止卡死
CREATE_GIF = True # 是否生成GIF
GIF_FILENAME = "evaluation_replay.gif"
OUTPUT_CSV_FILENAME = "evaluation_results.csv"

# Java 伺服器相關配置 (與訓練時一致)
JAVA_SERVER_JAR_PATH_EVAL = "/home/rvl/mingwei/NTUT_Deep_Learning/hw4/TetrisTCPserver_v0.6.jar"
JAVA_EXECUTABLE_PATH_EVAL = "/usr/bin/java"

def evaluate_model(model_path: str, n_envs: int, n_episodes: int, step_limit: int, create_gif: bool, gif_filename: str, csv_filename: str):
    """
    加載已訓練的模型並在 Tetris 環境中進行評估。
    """
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型文件於 {model_path}")
        return

    # 啟動 Java 伺服器 (如果尚未運行)
    # 在獨立評估腳本中，我們通常也需要確保伺服器已啟動
    # 為了避免與訓練腳本同時運行時的衝突，可以考慮檢查端口或傳遞參數
    print("評估腳本：嘗試啟動 Java Tetris 伺服器（如果需要）...")
    try:
        # 一個簡單的檢查，如果伺服器已在運行，Popen可能不會做任何事或Java程序會退出
        subprocess.Popen([JAVA_EXECUTABLE_PATH_EVAL, "-jar", JAVA_SERVER_JAR_PATH_EVAL])
        time.sleep(2) # 給伺服器一點啟動時間
    except Exception as e:
        print(f"評估腳本：啟動 Java 伺服器失敗: {e}")
        # 根據情況決定是否繼續，如果伺服器必須由這個腳本啟動

    # 創建評估環境
    print(f"正在創建 {n_envs} 個並行環境用於評估...")
    eval_vec_env = make_vec_env(lambda: TetrisEnv(), n_envs=n_envs, vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv)
    print("評估環境創建完成。")

    # 加載模型
    print(f"正在從 {model_path} 加載模型...")
    try:
        model = PPO.load(model_path, env=eval_vec_env)
        print("模型加載成功。")
    except Exception as e:
        print(f"加載模型失敗: {e}")
        eval_vec_env.close()
        return

    all_ep_rewards, all_ep_lengths, all_ep_lines_removed = [], [], []
    max_reward_overall = -float('inf')
    best_game_frames_overall = []
    max_lines_overall = 0
    max_steps_overall = 0

    episodes_completed = 0
    current_episode_rewards = np.zeros(n_envs)
    current_episode_lengths = np.zeros(n_envs, dtype=int)
    current_episode_frames = [[] for _ in range(n_envs)]

    obs = eval_vec_env.reset()

    print(f"開始評估，目標運行 {n_episodes} 個回合...")
    while episodes_completed < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_vec_env.step(action)

        for i in range(n_envs):
            current_episode_rewards[i] += rewards[i]
            current_episode_lengths[i] += 1

            # 收集幀用於GIF
            frame_to_save = None
            current_obs_for_env = None
            if isinstance(obs, list) and len(obs) > i: current_obs_for_env = obs[i]
            elif isinstance(obs, np.ndarray) and obs.ndim >=1 and obs.shape[0] > i: current_obs_for_env = obs[i]

            if current_obs_for_env is not None and \
               current_obs_for_env.shape == (TetrisEnv.IMG_HEIGHT, TetrisEnv.IMG_WIDTH, TetrisEnv.IMG_CHANNELS):
                frame_to_save = current_obs_for_env.copy()

            if frame_to_save is not None and create_gif:
                current_episode_frames[i].append(frame_to_save)

            if dones[i] or current_episode_lengths[i] >= step_limit:
                episodes_completed += 1
                all_ep_rewards.append(current_episode_rewards[i])
                all_ep_lengths.append(current_episode_lengths[i])
                lines_this_ep = infos[i].get('removed_lines', 0)
                all_ep_lines_removed.append(lines_this_ep)

                print(f"回合 {episodes_completed}/{n_episodes} (環境 {i}) 完成: "
                      f"獎勵={current_episode_rewards[i]:.2f}, "
                      f"長度={current_episode_lengths[i]}, "
                      f"消除行數={lines_this_ep}")

                if current_episode_rewards[i] > max_reward_overall:
                    max_reward_overall = current_episode_rewards[i]
                    if create_gif:
                        best_game_frames_overall = list(current_episode_frames[i])
                    max_lines_overall = lines_this_ep
                    max_steps_overall = current_episode_lengths[i]

                # 重置該環境的追蹤變量 (注意：VecEnv 會自動 reset)
                current_episode_rewards[i] = 0
                current_episode_lengths[i] = 0
                current_episode_frames[i] = []

                if episodes_completed >= n_episodes:
                    break # 跳出內層循環
        if episodes_completed >= n_episodes:
            break # 跳出外層循環 (步數循環)


    print("\n--- 評估結果總結 ---")
    if all_ep_rewards:
        print(f"平均獎勵: {np.mean(all_ep_rewards):.2f} +/- {np.std(all_ep_rewards):.2f}")
        print(f"平均回合長度: {np.mean(all_ep_lengths):.2f}")
        print(f"平均消除行數: {np.mean(all_ep_lines_removed):.2f}")
        print(f"評估中最佳遊戲: 獎勵={max_reward_overall:.2f}, 消除行數={max_lines_overall}, 步數={max_steps_overall}")
    else:
        print("評估中沒有完成任何回合。")
        max_lines_overall, max_steps_overall = 0,0 # 確保有默認值

    if create_gif and best_game_frames_overall:
        print(f"正在生成GIF動畫: {gif_filename}...")
        try:
            imageio.mimsave(gif_filename, best_game_frames_overall, loop=0, duration=100)
            print(f"GIF動畫已保存到 {gif_filename}")
        except Exception as e:
            print(f"保存GIF失敗: {e}")
    elif create_gif:
        print("沒有找到最佳遊戲的幀來生成GIF。")

    print(f"正在將結果寫入CSV: {csv_filename}...")
    with open(csv_filename, 'w') as fs:
        fs.write('id,removed_lines,played_steps\n')
        fs.write(f'0,{max_lines_overall},{max_steps_overall}\n') # Kaggle 通常需要 id 0 和 1
        fs.write(f'1,{max_lines_overall},{max_steps_overall}\n')
    print(f"結果已寫入 {csv_filename}")

    eval_vec_env.close()
    print("評估環境已關閉。")

if __name__ == "__main__":
    # --- 配置 logging (如果作為獨立腳本運行) ---
    eval_log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=eval_log_format)

    # 確保輸出文件夾存在 (如果GIF或CSV保存在子目錄中)
    # os.makedirs("./evaluation_output/", exist_ok=True)

    evaluate_model(
        model_path=MODEL_PATH,
        n_envs=N_EVAL_ENVS,
        n_episodes=N_EVAL_EPISODES,
        step_limit=TEST_STEPS_PER_EPISODE_LIMIT,
        create_gif=CREATE_GIF,
        gif_filename=GIF_FILENAME,
        csv_filename=OUTPUT_CSV_FILENAME
    )
