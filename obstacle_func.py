import cv2
import numpy as np
import math

# コマンド送信用の関数
def send_motor_command(ser, command):
    ser.write(command.encode('utf-8'))
    ser.flush()
    print(f"Sending command: {command}")
    
# 距離測定と段差・障害物の検出を行う関数
# 引数 depth は depth_scale (例: 0.001) を想定しています
def process_step(depth_img, color_img,camera, depth_start_x, depth_end_x, depth_h, depth_w):
    
    # 画像データがない場合はNoneを返す
    if depth_img is None:
        return None, False

    # --- 1. 前処理: 中央短冊の切り出しと1/zプロファイルの作成 ---
    strip_data = depth_img[:, depth_start_x:depth_end_x]
    
    # 行ごとの平均距離を計算し、メートル単位に変換
    row_means = np.mean(strip_data, axis=1) * camera.depth_scale
    
    # 1/z プロファイル作成 (近すぎるノイズ除去含む)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_means[row_means < 0.1] = np.nan # 近すぎる値はノイズとして除外
        inv_z_profile = 1.0 / row_means
        # NaNやInfを0に置換
        inv_z_profile = np.nan_to_num(inv_z_profile, nan=0.0, posinf=0.0, neginf=0.0)

    # 平滑化 (移動平均フィルタ: kernel_size=15)
    kernel_size = 15
    kernel = np.ones(kernel_size) / kernel_size
    inv_z_smooth = np.convolve(inv_z_profile, kernel, mode='same')
    
    # --- 2. 障害物検出ロジック (detect_step.pyより移植) ---
    obstacle_detected_start = None # 障害物の始まり（足元側Y座標）
    obstacle_detected_end = None   # 障害物の終わり（頭側Y座標）
    h = depth_h # 画像の高さ##################################これ何に使用？

    # (A) 足元の勾配（Ground Slope）を学習する
    # 画像下部(h-20〜h-40)を使って基準となる地面の傾きを計算
    base_slope_list = []
    check_base_start = h - 20
    check_base_end = h - 40
    
    for k in range(check_base_start, check_base_end, -1):
        # インデックス範囲外参照防止
        """
        k_idx = max(0, min(h-1, k))
        prev_idx = max(0, min(h-1, k+5))
        val_k = inv_z_smooth[k_idx]
        val_prev = inv_z_smooth[prev_idx]
        """
        val_k = inv_z_smooth[k]
        val_prev = inv_z_smooth[min(depth_h-1, k+5)]
        slope = val_prev - val_k # 手前の方が1/zが大きいので正の値になるはず
        
        if val_k > 0.5: # データが有効な場合のみ追加
            base_slope_list.append(slope)
    
    # 基準勾配 (計算できなければデフォルト0.05)
    ground_slope = np.mean(base_slope_list) if base_slope_list else 0.05
    
    # (B) 障害物探索ループ
    scan_start = h - 45
    scan_end = h // 3 # 画面の上1/3まで探索
    
    obstacle_pixel_count = 0     # 壁らしきピクセルの連続数
    REQUIRED_HEIGHT_PIXELS = 15  # 何ピクセル続いたら障害物とみなすか
    
    for y in range(scan_start, scan_end, -1):
        """
        y_idx = max(0, min(h-1, y))
        prev_idx = max(0, min(h-1, y+5))
        val = inv_z_smooth[y_idx]
        prev_val = inv_z_smooth[prev_idx]
        """
        val = inv_z_smooth[y]
        prev_val = inv_z_smooth[min(depth_h-1, y+5)]
        
        # 勾配（差分）計算
        current_diff = prev_val - val
        
        # ■判定ロジック■
        # 変化が地面の傾きの30%未満かつ、ある程度近い(val>0.5)場合は「壁」とみなす
        is_wall = (current_diff < ground_slope * 0.3) and (val > 0.5)
        
        if is_wall:
            obstacle_pixel_count += 1
        else:
            # 壁が途切れた時、高さが十分にあれば検出確定
            if obstacle_pixel_count > REQUIRED_HEIGHT_PIXELS:
                obstacle_detected_start = y + obstacle_pixel_count
                obstacle_detected_end = y
                break # 最初に見つかった障害物を採用してループ終了
            
            # 高さ不足ならノイズとしてリセット
            obstacle_pixel_count = 0
            
    # ループ終了後に壁が続いていた場合の処理
    if obstacle_detected_start is None and obstacle_pixel_count > REQUIRED_HEIGHT_PIXELS:
        obstacle_detected_start = scan_end + obstacle_pixel_count
        obstacle_detected_end = scan_end

    # --- 3. 結果画像の生成 ---
    # Depth画像を可視化用にカラーマップ変換 (alphaは表示の明るさ調整)
    #depth_vis = cv2.convertScaleAbs(depth_img, alpha=0.03)
    #result_img = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    # 解析領域（緑枠）
    cv2.rectangle(color_img, (depth_start_x, 0), (depth_end_x, h), (0, 255, 0), 1)
    
    is_obstacle = False
    if obstacle_detected_start is not None:
        is_obstacle = True
        # 障害物エリア（赤枠）
        cv2.rectangle(color_img, 
                      (depth_start_x, obstacle_detected_end), 
                      (depth_end_x, obstacle_detected_start), 
                      (0, 0, 255), 2)
        cv2.putText(color_img, "OBSTACLE", (depth_start_x, obstacle_detected_start - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return is_obstacle, color_img


def process_horizontal_obstacle(depth_img, color_img, camera,y_start,y_end):
    """
    カメラ画像中央に横長の領域を設定し、距離を測定。
    X軸方向に走査し、「近い距離」かつ「平坦（変化が少ない）」な領域を障害物として検出する。
    """
    if depth_img is None:
        return "NONE", color_img
    
    h,w = depth_img.shape
    
    # 短冊切り出し
    strip_data = depth_img[y_start:y_end, :] # 横幅は全域
    
    # 2. Y軸方向（縦）につぶして、X軸ごとの平均距離を計算
    # 行方向(axis=0)に平均をとる -> 長さwの1次元配列になる
    col_means = np.mean(strip_data, axis=0) * camera.depth_scale
    
    # ノイズ処理: 距離0または極端に遠い/近い値をフィルタリング
    # ここでは単純化のため、0を大きな値（無視）に置換
    col_means[col_means < 0.1] = 10.0 
    
    # 3. X軸方向の変化量（微分に近いもの）を計算
    # np.diff で隣り合う要素との差分をとる
    # diff[i] = col_means[i+1] - col_means[i]
    diff = np.abs(np.diff(col_means))
    # diff配列は長さがw-1になるため、最後に0を追加してサイズを合わせる
    diff = np.append(diff, 0)

    # 4. 障害物判定ロジック
    # 条件A: 距離が近い (例: 1.0m以内)
    # 条件B: 距離の変化が小さい (平坦である = 物体の表面)
    
    THRESHOLD_DIST = 1.0  # [m] これより近いと障害物候補
    THRESHOLD_FLAT = 0.05 # [m] 隣との差がこれ以下なら「平坦」とみなす
    MIN_WIDTH_PIXELS = 40 # [px] この幅以上条件を満たせば障害物と認定
    
    obstacle_count = 0
    start_x = None
    detected_obstacles = [] # (start_x, end_x, avg_dist)

    for x in range(w):
        dist = col_means[x]
        change = diff[x]
        
        # 「近くて」かつ「平坦」か？
        if dist < THRESHOLD_DIST and change < THRESHOLD_FLAT:
            if obstacle_count == 0:
                start_x = x
            obstacle_count += 1
        else:
            # 条件を満たさなくなった時、これまで蓄積した幅が十分か判定
            if obstacle_count > MIN_WIDTH_PIXELS:
                # 障害物確定
                end_x = x
                avg_d = np.mean(col_means[start_x:end_x])
                detected_obstacles.append((start_x, end_x, avg_d))
            
            # リセット
            obstacle_count = 0
            start_x = None

    # 画面右端まで続いていた場合の処理
    if obstacle_count > MIN_WIDTH_PIXELS:
         detected_obstacles.append((start_x, w, np.mean(col_means[start_x:w])))

    # 5. 結果の描画とステータス返却
    # 解析領域（青枠）
    cv2.rectangle(color_img, (0, y_start), (w, y_end), (255, 255, 0), 1)
    
    status = False
    target_center = None
    
    # 最も近い障害物を採用、または一番大きい障害物を採用するなど
    if detected_obstacles:
        status = True
        # 例: 一番近い障害物をターゲットにする
        best_obs = min(detected_obstacles, key=lambda o: o[2]) 
        ox_start, ox_end, ox_dist = best_obs
        
        # 赤枠で囲む
        cv2.rectangle(color_img, (ox_start, y_start), (ox_end, y_end), (0, 0, 255), 2)
        cv2.putText(color_img, f"Obs {ox_dist:.2f}m", (ox_start, y_start - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 障害物の中心座標
        target_center = (ox_start + ox_end) // 2

    return status, color_img