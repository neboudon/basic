import cv2
import pyrealsense2 as rs
import threading
import time
import numpy as np
import serial

#シリアル通信用パラメータ
SERIAL_PORT = '/dev/ttyS0' 
SERIAL_BAUDRATE = 921600  # Pico側もこれに合わせてください

from detect_3way import send_motor_command, detect_step_or_obstacle

#カメラ画像取得用のクラス
class CameraStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.capture_time = 0
        color_w, color_h = 640, 480
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
        

def main():
    camera = CameraStream()
    camera.start()
    
    #処理レート
    INTERVAL_STEP = 0.500 # 2Hz
    
    #ループ全体の処理時間
    LOOP_PERIOD = 0.025 # 40Hzの基本周期
    
    #時間管理変数
    last_time_step = 0
    last_time_obst = 0
    
    while True:
        first_color_frame, first_depth_frame, _ = camera.get_latest()
        if first_depth_frame is None or first_color_frame is None:
            print("カメラの起動中...")
            time.sleep(0.1)
            continue
        break
    
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
    
    while True:
        loop_start_time = time.time()
        
        #フラグの初期化
        send_command_flag = False
        emergency_flag = False
        
        #画像の取得
        color_img ,depth_img, img_time = camera.get_latest()
        
        if(loop_start_time - img_time) > 0.1:
            print("警告:映像遅延．停止します．")
            emergency_flag = True
            
        if(loop_start_time - last_time_step) >= INTERVAL_STEP:
            #段差と障害物検出の実行
            #result , result_img = process_step(depth_img,color_img,camera,depth_start_x,depth_end_x,depth_h,depth_w)
            result, result_img = detect_step_or_obstacle(depth_img,color_img,camera)
            last_time_step = loop_start_time
            
            if result == "STEP":
                final_cmd = "S\n"
                send_command_flag = True
            elif result == "OBSTACLE":
                final_cmd = "avoid"
                send_command_flag = True
            else:
                send_command_flag = False
            """        
        if(loop_start_time - last_time_obst) >= INTERVAL_STEP:
            result, result_img = process_horizontal_obstacle(depth_img,color_img,camera,y_start,y_end,x_start,x_end)
            last_time_obst = loop_start_time
            
            if result == True:
                final_cmd = "S\n"
                send_command_flag = True
            else:
                send_command_flag = False
            """
        else:
            pass
        
        cv2.imshow("result_img",result_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
        if emergency_flag == True:
            final_cmd = "S\n"
            send_motor_command(ser,final_cmd)
            flow_result = None
        elif send_command_flag == True:
            send_motor_command(ser,final_cmd)
            flow_result = final_cmd
            
        elapsed = time.time() - loop_start_time
        sleep_time = LOOP_PERIOD -elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()