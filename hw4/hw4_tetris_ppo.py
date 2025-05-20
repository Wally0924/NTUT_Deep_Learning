import numpy as np
import socket
import cv2
import subprocess
import gymnasium as gym
from gymnasium import spaces
import torch
import os
import time
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# --- 全局配置 ---
JAVA_SERVER_JAR_PATH = "/home/rvl/mingwei/NTUT_Deep_Learning/hw4/TetrisTCPserver_v0.6.jar"
JAVA_EXECUTABLE_PATH = "/usr/bin/java"
HOST_IP = "127.0.0.1"
HOST_PORT = 10612
N_ENVS = 64
TOTAL_TIMESTEPS_TRAIN = 3_000_000
EVAL_FREQ_PER_ENV = 100000 # EvalCallback 的評估頻率

# --- 配置 Logging ---
log_format = '%(asctime)s - %(name)s - PID:%(process)d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
root_logger = logging.getLogger()
env_logger = logging.getLogger("TetrisEnv")

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    N_DISCRETE_ACTIONS = 5
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 200, 100, 3
    _global_server_connection_logged = False

    def __init__(self, host_ip=HOST_IP, host_port=HOST_PORT):
        super().__init__()
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        self.server_ip, self.server_port = host_ip, host_port
        self.client_sock = None
        self.pid = os.getpid()
        self._connect_server()
        self._reset_internal_state()

    def _reset_internal_state(self):
        self.current_observation = self._default_observation()
        self.total_lines_removed_in_episode = 0 # 追蹤本回合已消除的總行數
        self.server_reported_total_lines = 0 # 伺服器報告的累計總行數（可能跨回合）
        self.current_holes = 0
        self.current_max_height = 0
        self.current_lifetime = 0

    def _connect_server(self):
        if self.client_sock:
            try: self.client_sock.close()
            except Exception: pass
        try:
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_sock.settimeout(10.0)
            self.client_sock.connect((self.server_ip, self.server_port))
            if not TetrisEnv._global_server_connection_logged:
                root_logger.info(f"主要連接成功：至少一個 TetrisEnv 實例已連接到伺服器 {self.server_ip}:{self.server_port}")
                TetrisEnv._global_server_connection_logged = True
        except Exception as e:
            env_logger.error(f"PID {self.pid}: 連接伺服器時發生錯誤: {e}")
            raise

    def _get_board_metrics_from_server_response(self, server_height, server_holes):
        self.current_max_height = server_height
        self.current_holes = server_holes

    def step(self, action):
        if self.client_sock is None: self._connect_server()
        command_map = {0: b"move -1\n", 1: b"move 1\n", 2: b"rotate 0\n", 3: b"rotate 1\n", 4: b"drop\n"}
        command = command_map.get(action)
        if command is None: return self.current_observation, 0.0, False, False, {}

        try:
            self.client_sock.sendall(command)
            # server_total_lines 是伺服器自啟動以來累計消除的總行數
            terminated, server_cumulative_total_lines, server_height, server_holes, observation = self._get_server_response()
            self.current_observation = observation
            self._get_board_metrics_from_server_response(server_height, server_holes)

            reward = 0.0
            # 計算本次 step 實際消除的行數
            # self.server_reported_total_lines 是上一步伺服器報告的總行數
            lines_cleared_this_step = server_cumulative_total_lines - self.server_reported_total_lines
            self.server_reported_total_lines = server_cumulative_total_lines # 更新伺服器報告的總行數
            self.total_lines_removed_in_episode += lines_cleared_this_step # 更新本回合消除的行數

            # 1. 生存獎勵
            if not terminated:
                reward += 1.0 # 每一步給予固定的小獎勵以鼓勵生存

            # 2. 消除行的巨大獎勵
            if lines_cleared_this_step > 0:
                if lines_cleared_this_step == 1: reward += 500.0
                elif lines_cleared_this_step == 2: reward += 1500.0
                elif lines_cleared_this_step == 3: reward += 3000.0
                elif lines_cleared_this_step >= 4: reward += 5000.0

            # 3. 遊戲結束的懲罰
            if terminated:
                reward -= 200.0 # 遊戲結束給予負獎勵

            # 4. 暫時移除或極輕對洞和高度的懲罰
            # reward -= self.current_holes * 0.01
            # reward -= (self.current_max_height / 20.0) * 0.1

            self.current_lifetime += 1
            info = {'removed_lines': self.total_lines_removed_in_episode,
                    'lifetime': self.current_lifetime,
                    'is_success': terminated, # 用於 EvalCallback
                    'lines_this_step': lines_cleared_this_step # 添加這個信息用於調試
                   }
            return self.current_observation, reward, terminated, False, info
        except (BrokenPipeError, ConnectionResetError, socket.error, AttributeError) as e:
            env_logger.warning(f"PID {self.pid}: step() 網路/Socket 錯誤: {e}. 返回終止並嘗試重置。")
            self.close()
            return self._default_observation(), -500.0, True, False, {}
        except Exception as e:
            env_logger.error(f"PID {self.pid}: step() 發生未知錯誤: {e}")
            return self._default_observation(), -500.0, True, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.client_sock is None: self._connect_server()
        try:
            self.client_sock.sendall(b"start\n")
            # 伺服器在 reset (start) 後返回的 lines, height, holes 應為初始狀態
            _, server_cumulative_total_lines_at_reset, server_height, server_holes, observation = self._get_server_response()
            self._get_board_metrics_from_server_response(server_height, server_holes)
            # 初始化 server_reported_total_lines 為伺服器在遊戲開始時的總行數
            self.server_reported_total_lines = server_cumulative_total_lines_at_reset
        except (BrokenPipeError, ConnectionResetError, socket.error, AttributeError) as e:
            env_logger.warning(f"PID {self.pid}: reset() 網路/Socket 錯誤: {e}. 嘗試重新連接。")
            self.close(); self._connect_server()
            try:
                self.client_sock.sendall(b"start\n")
                _, server_cumulative_total_lines_at_reset, server_height, server_holes, observation = self._get_server_response()
                self._get_board_metrics_from_server_response(server_height, server_holes)
                self.server_reported_total_lines = server_cumulative_total_lines_at_reset
            except Exception as e_retry:
                env_logger.error(f"PID {self.pid}: reset() 重試連接後仍然失敗: {e_retry}"); observation = self._default_observation()
                self.server_reported_total_lines = 0 # 假設錯誤時為0
        except Exception as e:
            env_logger.error(f"PID {self.pid}: reset() 發生未知錯誤: {e}"); observation = self._default_observation()
            self.server_reported_total_lines = 0 # 假設錯誤時為0

        self._reset_internal_state() # 重置回合特定狀態
        self.current_observation = observation if observation is not None else self._default_observation()
        return self.current_observation, {}

    def _default_observation(self):
        return np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)

    def render(self): pass

    def close(self):
        if self.client_sock:
            try: self.client_sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            self.client_sock.close(); self.client_sock = None

    def _get_server_response(self):
        if self.client_sock is None: raise ConnectionAbortedError("Socket 未初始化")
        try:
            self.client_sock.settimeout(15.0)
            def recv_all(sock, num_bytes):
                data = b''; received_len = 0
                while received_len < num_bytes:
                    try: packet = sock.recv(num_bytes - received_len)
                    except socket.timeout: raise socket.timeout(f"recv_all 超時, 已接收 {received_len}/{num_bytes}")
                    if not packet: raise ConnectionAbortedError(f"Socket 在接收 {num_bytes} 字節時關閉")
                    data += packet; received_len = len(data)
                return data
            is_game_over = (recv_all(self.client_sock, 1) == b'\x01')
            # 這個 removed_lines 是伺服器自啟動以來累計消除的總行數
            cumulative_total_lines_from_server = int.from_bytes(recv_all(self.client_sock, 4), 'big')
            height = int.from_bytes(recv_all(self.client_sock, 4), 'big')
            holes = int.from_bytes(recv_all(self.client_sock, 4), 'big')
            img_size = int.from_bytes(recv_all(self.client_sock, 4), 'big')
            np_image = self._default_observation()
            if img_size < 0: env_logger.warning(f"PID {self.pid}: 伺服器發送了無效的圖像大小 {img_size}")
            elif img_size > 0:
                img_png = recv_all(self.client_sock, img_size)
                if img_png:
                    nparr = np.frombuffer(img_png, np.uint8)
                    if nparr.size > 0:
                        decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if decoded_image is not None and decoded_image.shape == (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS):
                            np_image = decoded_image
            self.client_sock.settimeout(None)
            return is_game_over, cumulative_total_lines_from_server, height, holes, np_image
        except socket.timeout:
            env_logger.error(f"PID {self.pid}: 從伺服器接收數據超時。")
            return True, self.server_reported_total_lines, self.current_max_height, self.current_holes, self._default_observation()
        except (ConnectionAbortedError, ConnectionResetError, socket.error) as e:
            env_logger.error(f"PID {self.pid}: 與伺服器的連接中斷 - {e}")
            return True, self.server_reported_total_lines, self.current_max_height, self.current_holes, self._default_observation()
        except Exception as e:
            env_logger.error(f"PID {self.pid}: _get_server_response 發生未知錯誤: {e}")
            return True, self.server_reported_total_lines, self.current_max_height, self.current_holes, self._default_observation()

def main():
    root_logger.info(f"主進程 PID: {os.getpid()}。嘗試啟動 Java Tetris 伺服器...")
    try:
        if not os.path.exists(JAVA_SERVER_JAR_PATH):
            root_logger.error(f"錯誤：找不到 Java 伺服器 JAR 文件於 {JAVA_SERVER_JAR_PATH}")
            return
        subprocess.Popen([JAVA_EXECUTABLE_PATH, "-jar", JAVA_SERVER_JAR_PATH])
        time.sleep(3)
        root_logger.info("Java Tetris 伺服器啟動指令已發送。")
    except Exception as e:
        root_logger.error(f"啟動 Java 伺服器失敗: {e}")
        return

    from stable_baselines3.common.env_checker import check_env
    root_logger.info("正在創建單個環境進行檢查...")
    try:
        check_env(TetrisEnv())
        root_logger.info("環境檢查通過。")
    except Exception as e:
        root_logger.error(f"環境檢查失敗: {e}"); return

    root_logger.info(f"正在創建 {N_ENVS} 個並行環境 (使用 SubprocVecEnv)...")
    vec_env_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
    train_vec_env = make_vec_env(lambda: TetrisEnv(), n_envs=N_ENVS, vec_env_cls=vec_env_cls)
    root_logger.info("訓練用並行環境創建完成。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        root_logger.info(f"檢測到CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        root_logger.info("未檢測到CUDA GPU，將使用CPU。")

    # --- 調整後的模型超參數 (PPO) ---
    model_params = {
        'policy': "CnnPolicy",
        'env': train_vec_env,
        'learning_rate': 5e-4,      # 嘗試稍高的學習率
        'n_steps': 512,             # 增加 n_steps
        'batch_size': 64,           # 減小 batch_size
        'n_epochs': 4,              # 減少 n_epochs
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.1,            # 大幅增加熵係數
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'verbose': 1,
        'device': device,
        'tensorboard_log': "./sb3_log_survival_focus_v2/" # 更新日誌目錄
    }
    model = PPO(**model_params)

    # --- 設置 EvalCallback ---
    eval_log_path = './logs_survival_focus_v2/'
    os.makedirs(os.path.join(eval_log_path, 'best_model/'), exist_ok=True)
    os.makedirs(os.path.join(eval_log_path, 'results/'), exist_ok=True)

    eval_env_for_callback = make_vec_env(lambda: TetrisEnv(), n_envs=max(1, N_ENVS // 4), vec_env_cls=vec_env_cls)
    eval_callback = EvalCallback(eval_env_for_callback,
                                 best_model_save_path=os.path.join(eval_log_path, 'best_model/'),
                                 log_path=os.path.join(eval_log_path, 'results/'),
                                 eval_freq=max(1, EVAL_FREQ_PER_ENV),
                                 n_eval_episodes=10, # 評估時運行的回合數
                                 deterministic=True, render=False, warn=False)

    root_logger.info(f"開始訓練 PPO 模型，總步數: {TOTAL_TIMESTEPS_TRAIN}...")
    start_time_train = time.time()
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS_TRAIN, progress_bar=True, callback=eval_callback)
    except Exception as e:
        root_logger.error(f"訓練過程中發生錯誤: {e}")
        model.save("tetris_ppo_model_survival_focus_v2_interrupted.zip")
    finally:
        train_vec_env.close()
        eval_env_for_callback.close()

    end_time_train = time.time()
    training_duration_seconds = end_time_train - start_time_train
    root_logger.info(f"模型訓練完成。總耗時: {training_duration_seconds:.2f} 秒 ({training_duration_seconds/3600:.2f} 小時)")

    model_save_path = "tetris_ppo_model_survival_focus_v2_final.zip"
    model.save(model_save_path)
    root_logger.info(f"最終模型已保存為 {model_save_path}")
    best_model_final_path = os.path.join(eval_log_path, 'best_model/best_model.zip')
    if os.path.exists(best_model_final_path):
        root_logger.info(f"最佳模型保存在: {best_model_final_path}")
    else:
        root_logger.warning("EvalCallback 未能成功保存最佳模型。")


if __name__ == '__main__':
    os.makedirs("./sb3_log_survival_focus_v2/", exist_ok=True)
    main()
