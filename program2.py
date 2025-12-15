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
from func2 import send_motor_command, process_cog, process_mis ,detect_step_or_obstacle ,process_wall_distance ,calc_avoidance_command

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
    
    #障害物回避のための直進時間
    KEEP_TIME = 3.00
    
    #時間管理変数
    last_time_cog = 0
    last_time_step = 0
    last_time_avoid = 0
    time_to_forward = 0
    
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
            
        
        if((loop_start_time - last_time_cog) >= INTERVAL_COG) and(avoid_flag == False):
            
            #重心検出の実行
            #final_cmd = process_cog(color_img,resize_h)
            #final_cmd,result_img = process_cog(color_img,resize_h)
            
            #消失点検出の実行
            #final_cmd = process_mis(color_img, resize_h, clahe)
            final_cmd,result_img = process_mis(color_img, resize_h, clahe)
            
            if recovering_flag == True and final_cmd == "F\n":
                recovering_flag = False
                print("復帰完了：段差検出を再開します")
            
            #フラグの更新
            last_time_cog = loop_start_time #最終処理時間の更新
            send_command_flag = True #コマンド送信のフラグを立てる
        
        #if((loop_start_time - last_time_step) >= INTERVAL_DETECT) and prev_cmd == "F\n" and avoid_flag == False:
        #上の開始条件は変更することを考える #直進だけに絞らないが、ロボットが回避をした時に、戻ってくる時の誤検出ができるようにする
        if((loop_start_time - last_time_step) >= INTERVAL_DETECT) and avoid_flag == False and recovering_flag == False:
            #pass
            result,result_img = detect_step_or_obstacle(depth_img,color_img,camera)
            last_time_step = loop_start_time
            if result == "STEP":
                final_cmd = "S\n"
                send_command_flag = True
            elif result == "OBSTACLE":
                avoid_flag = True
                wall_side = 'right'
            #else:
            #    send_command_flag = False
            
        if((avoid_flag == True) and (loop_start_time - last_time_avoid) >= INTERVAL_AVOID):
            dist = process_wall_distance(depth_img,camera,roi_y1,roi_y2,roi_w,depth_h,depth_w,wall_side)
            last_time_avoid = loop_start_time #時刻の更新
            
            final_cmd = calc_avoidance_command(dist,wall_side)
            send_command_flag = True
            
            error = dist - TARGET_DISTANCE
            if keep_flag == False:
                #壁に接近中
                #コマンド作成
                #final_cmd = calc_avoidance_command(dist,wall_side)
                #send_command_flag = True
                if abs(error) < ERROR_THRESHOLD:
                    keep_flag = True
                    time_to_forward = loop_start_time
                    print("[Avoid] Target Distance Reached. Keeping for 3s...")

                
            else:
                if((loop_start_time - time_to_forward) >= KEEP_TIME):
                    avoid_flag = False
                    keep_flag = False
                    recovering_flag = True
                    print("回避終了：復帰モードへ移行")
            #コマンド作成
            #final_cmd = calc_avoidance_command(dist,wall_side)
            #send_command_flag = True
        
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