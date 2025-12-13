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

#func1からの関数の呼び出し
from func1 import send_motor_command, process_cog, process_mis

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
    INTERVAL_COG = 0.5
    
    #ループ全体の処理時間
    LOOP_PERIOD = 0.025 # 40Hzの基本周期
    
    #消失点検出のためのパラメータ
    CLIP_LIMIT = 15.0
    TILE_GRID_SIZE = (4, 4)
    
    #時間管理変数
    last_time_cog = 0
    
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
    
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
        print(f"[メイン]: ポート {SERIAL_PORT} (Baud: {SERIAL_BAUDRATE}) を開きました。")
        time.sleep(2)  # シリアルポートの初期化待ち
    
    except serial.SerialException as e:
        print("シリアルポートのオープンに失敗しました。")
        return
    
    print("制御を開始します")
    
    #変数の初期化
    flow_result = None
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    
    result_img = np.zeros((resize_h, RESIZE_WIDTH, 3), dtype=np.uint8)
    
    while True:
        loop_start_time = time.time()
        
        #フラグの初期化
        send_command_flag = False #コマンドを送るか？
        emergency_flag = False #カメラのフレームが遅れているか？
        
        #画像の取得
        color_img, depth_img , img_time = camera.get_latest()
        
        if(loop_start_time - img_time) > 0.1:
            print("警告:映像遅延．停止します．")
            emergency_flag = True
            
        
        if((loop_start_time - last_time_cog) >= INTERVAL_COG):
            
            #重心検出の実行
            #final_cmd = process_cog(color_img,resize_h)
            final_cmd,result_img = process_cog(color_img,resize_h)
            
            #消失点検出の実行
            #final_cmd = process_mis(color_img, resize_h, clahe)
            final_cmd,result_img = process_mis(color_img, resize_h, clahe)
            
            #フラグの更新
            last_time_cog = loop_start_time #最終処理時間の更新
            send_command_flag = True #コマンド送信のフラグを立てる
            
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
            flow_result = None
            
        elif send_command_flag == True:
            #コマンド送信
            send_motor_command(ser,final_cmd)
            flow_result = final_cmd
        
        elapsed = time.time() - loop_start_time
        sleep_time = LOOP_PERIOD - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()