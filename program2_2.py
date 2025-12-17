#このプログラムは2Hzで消失点もしくは重心を検出してロボットが排水路内の中心を走行できるように操舵する
#1Hzでロボット正面の距離を測定して，正面に障害物や段差がないかを判定するプログラムである
#障害物を検出したら障害物回避を行う
#正面にのみ障害物がある場合は右90回転，直進，左90回転，の3通りの回避動作を行う
#正面と左に障害物がある場合は右90回転，直進，左90回転，の3通りの回避動作を行う
#正面と右に障害物がある場合は左90回転,直進，右90回転，の3通りの回避動作を行う
#過去に障害物回避を右回避で行ったら，次は左回避を優先的に行う
#ただし，一度直進コマンドを送ったらリセットする
#すべての場所で障害物が検出された場合は停止する
#回避を行った後は一度消失点を検出してから段差検出を再開する
import cv2
import pyrealsense2 as rs
import threading
import time
import numpy as np
import serial

#重心と消失点検出用のパラメータ
RESIZE_WIDTH = 240

#シリアル通信用パラメータ
SERIAL_PORT = '/dev/ttyS0' 
SERIAL_BAUDRATE = 921600  # Pico側もこれに合わせてください

#func2からの関数の呼び出し
from func2_2 import send_motor_command, process_cog, process_mis ,detect_step_or_obstacle ,process_wall_distance ,calc_avoidance_command

#カメラ画像取得用のクラス
class CameraStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.capture_time = 0
        color_w, color_h = 640, 360
        depth_w, depth_h = 640, 480
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, 60)
        self.config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, 60)
        profile = self.pipeline.start(self.config)
        self.depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.capture_time = 0
        
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        
    def update(self):
        while True:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            now = time.time()
            
            with self.lock:
                self.latest_color_frame = color_image
                self.latest_depth_frame = depth_image
                self.capture_time = now
    
    def get_latest(self):
        with self.lock:
            return self.latest_color_frame, self.latest_depth_frame, self.capture_time
        
        
#メイン関数
def main():
    camera = CameraStream()
    camera.start()
    
    #処理レート(秒)
    INTERVAL_COG = 0.5 #1秒に2回
    INTERVAL_DETECT = 1 #1秒に1回
    INTERVAL_AVOID = 0.066 #15Hz
    
    #ループ全体の処理時間
    LOOP_PERIOD = 0.025 # 40Hzの基本周期
    
    #消失点検出のためのパラメータ
    CLIP_LIMIT = 15.0
    TILE_GRID_SIZE = (4, 4)
    
    #距離測定のパラメータ
    TARGET_DISTANCE = 0.8  # 目標距離 (m)
    ERROR_THRESHOLD = 0.1   # 許容誤差 (m)
    
    #障害物回避のためのフラグ
    avoid_flag = False
    keep_flag = False
    recovering_flag = False
    ctrl_flag = False
    prev_avoid_side = 'right'  # 前回どちらに避けたか
    reset_priority_flag = True # 直進したらリセットするフラグ
    force_cog_flag = False     # 回避後の復帰用フラグ
    target_avoid_cmd = ""      # 送信予定の回避コマンド
    
    #障害物回避のための直進時間
    KEEP_TIME = 3.00
    
    #時間管理変数
    last_time_cog = 0
    last_time_step = 0
    last_time_avoid = 0
    time_to_forward = 0
    
    #最初のフレームの取得
    while True:
        first_color_frame, first_depth_frame, _ = camera.get_latest()
        if first_depth_frame is None or first_color_frame is None:
            print("カメラの起動中...")
            time.sleep(0.1)
            continue
        break
    
    #重心と消失点検出用のリサイズの高さを計算
    orig_h, orig_w = first_color_frame.shape[:2]
    aspect_ratio = orig_h / orig_w
    resize_h = int(RESIZE_WIDTH * aspect_ratio)
    
    #段差検出のためのパラメータ計算
    depth_h, depth_w = first_depth_frame.shape[:2]
    roi_h = 100
    roi_w = 50
    roi_y1 = (depth_h // 2) - (roi_h // 2)
    roi_y2 = roi_y1 + roi_h
    
    #ポートのオープン
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
        print(f"[メイン]: ポート {SERIAL_PORT} (Baud: {SERIAL_BAUDRATE}) を開きました。")
        time.sleep(2)  # シリアルポートの初期化待ち
    
    except serial.SerialException as e:
        print("シリアルポートのオープンに失敗しました。")
        return
    
    print("制御を開始します")
    
    #変数の初期化
    prev_cmd = None
    wall_side = 'right'
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    
    result_img = np.zeros((resize_h, RESIZE_WIDTH, 3), dtype=np.uint8)
    
    while True:
        
        #もし前に送ったコマンドが停止コマンドなら停止コマンドを送り続ける
        if(prev_cmd == "S\n"):
            send_motor_command(ser,prev_cmd)
            time.sleep(0.1)
            continue
        
        loop_start_time = time.time()
        
        #フラグの初期化
        send_command_flag = False #コマンドを送るか？
        emergency_flag = False #カメラのフレームが遅れているか？
        
        #画像の取得
        color_img, depth_img , img_time = camera.get_latest()
        
        if(loop_start_time - img_time) > 0.1:
            print("警告:映像遅延．停止します．")
            emergency_flag = True
            
        #処理時間になっているかもしくは，障害物回避直後か
        #if((loop_start_time - last_time_cog) >= INTERVAL_COG) and(avoid_flag == False):
        if((loop_start_time - last_time_cog) >= INTERVAL_COG) or (force_cog_flag == True):
            
            #重心検出の実行
            #final_cmd = process_cog(color_img,resize_h)
            #final_cmd,result_img = process_cog(color_img,resize_h)
            
            #消失点検出の実行
            #final_cmd = process_mis(color_img, resize_h, clahe)
            final_cmd,result_img = process_mis(color_img, resize_h, clahe)
            
            #if recovering_flag == True and final_cmd == "F\n":
            if final_cmd == "F\n":
                reset_priority_flag = True
            #段差検出後なら段差検出を再開できるようにする
            if force_cog_flag == True:
                force_cog_flag = False # フラグを下ろして通常モードへ
                last_time_step = loop_start_time # 段差検出タイマーをリセット（即座に反応しないよう）
                print("復帰完了：通常走行に戻ります")
            
            #フラグの更新
            last_time_cog = loop_start_time #最終処理時間の更新
            send_command_flag = True #コマンド送信のフラグを立てる
        
        #if((loop_start_time - last_time_step) >= INTERVAL_DETECT) and prev_cmd == "F\n" and avoid_flag == False:
        #上の開始条件は変更することを考える #直進だけに絞らないが、ロボットが回避をした時に、戻ってくる時の誤検出ができるようにする
        #if((loop_start_time - last_time_step) >= INTERVAL_DETECT) and avoid_flag == False and recovering_flag == False:
        if((loop_start_time - last_time_step) >= INTERVAL_DETECT) and (force_cog_flag == False):
            #処理時間を経過しており，消失点の検出を行わないflagがfalseの時，検出を行う
            obs_status,result_img = detect_step_or_obstacle(depth_img,color_img,camera)
            last_time_step = loop_start_time
            if obs_status == "STEP":
                final_cmd = "S\n"
                send_command_flag = True
            
            elif obs_status != "SAFE":
                avoid_flag = True
                # --- 追加: 状況に応じた方向決定 ---
                if obs_status == "OBS_LEFT_CENTER":
                    target_avoid_cmd = "PR\n"       # 左+中は右へ
                elif obs_status == "OBS_RIGHT_CENTER":
                    target_avoid_cmd = "PL\n"       # 右+中は左へ
                elif obs_status == "OBS_CENTER":
                    # --- 追加: 交互動作ロジック ---
                    if reset_priority_flag == True:
                        target_avoid_cmd = "PR\n"   # 初回は右
                    else:
                        # 前回と逆方向へ
                        target_avoid_cmd = "PL\n" if prev_avoid_side == 'right' else "PR\n"
            #else:
            #    send_command_flag = False
        
        if (avoid_flag == True):
            send_motor_command(ser, target_avoid_cmd)
            if target_avoid_cmd == "PR\n":
                prev_avoid_side = 'right'
            elif target_avoid_cmd == "PL\n":
                prev_avoid_side = 'left'
    
            reset_priority_flag = False # 回避したのでリセット解除(次は交互動作)
                #send_motor_command(ser,"P\n")
            print("回避動作中...完了を待機します")
            original_timeout = ser.timeout
            ser.timeout = 20.0
            try:
                response = ser.readline()
                if response:
                    print(f"{response.decode('utf-8').strip()}")
                    avoid_flag = False
                    time.sleep(0.5)
                    last_time_step = time.time()
                    send_command_flag = False
                else:
                    print("タイムアウト: 返信なし")
                    avoid_flag = False
            except serial.SerialException as e:
                print(f"!!! 受信エラー: {e}")
            
            finally:
                ser.timeout = original_timeout
                avoid_flag = False
    
                # --- 追加: 復帰シーケンス ---
                force_cog_flag = True # 次回ループで消失点検出を強制実行
                send_command_flag = False
        else:
            pass
        
        cv2.imshow("result_img",result_img)
        key = cv2.waitKey(1)
        if key == 27: # ESCキーで終了
            break
        
        if emergency_flag == True:
            final_cmd = "S\n" #ここは修正する
            #コマンド送信
            send_motor_command(ser,final_cmd)
            prev_cmd = final_cmd
            
        elif send_command_flag == True:
            #コマンド送信
            send_motor_command(ser,final_cmd)
            prev_cmd = final_cmd
        
        elapsed = time.time() - loop_start_time
        sleep_time = LOOP_PERIOD - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()