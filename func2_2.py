import cv2
import numpy as np
import math

#コマンド送信用の関数
def send_motor_command(ser,command):
    ser.write(command.encode('utf-8'))
    ser.flush()
    print(f"Sending command: {command}")
    
#画像の暗点重心を検出する関数
def process_cog(color_image,resize_h):
    # 重心検出処理の変数の定義
    STEERING_THRESHOLD = 5  # 重心差の閾値（ピクセル）
    RESIZE_WIDTH = 240
    
    # リサイズとグレースケール変換
    resized_frame = cv2.resize(color_image, (RESIZE_WIDTH, resize_h), interpolation=cv2.INTER_AREA)
    height, width = resized_frame.shape[:2]
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    #暗点重心の検出
    inverted_array = 255 - gray_frame
    total_weight = np.sum(inverted_array)
    image_center_x = width // 2
    center_x = image_center_x
    if total_weight > 0:
        x_coords = np.arange(width)
        center_x = np.sum(x_coords * np.sum(inverted_array, axis=0)) / total_weight

    gravity_difference = center_x - image_center_x
    
    # 重心位置に赤い円を描画
    cv2.circle(resized_frame, (center_x, height // 2), 5, (0, 0, 255), -1)
    # 中心線を描画（緑）
    cv2.line(resized_frame, (image_center_x, 0), (image_center_x, height), (0, 255, 0), 1)
    
    steering_command = "F\n"
    
    if abs(gravity_difference) > STEERING_THRESHOLD:
        if gravity_difference > 0:
            steering_command = f"R {gravity_difference:.2f}\n" 
        else:
            steering_command = f"L {abs(gravity_difference):.2f}\n"
    
    #return steering_command
    return steering_command, resized_frame


#画像の線から消失点を求める関数
def process_mis(color_image, resize_h, clahe):
    # 重心検出処理の変数の定義
    STEERING_THRESHOLD = 5  # 重心差の閾値（ピクセル）
    RESIZE_WIDTH = 240
    # Cannyエッジ検出の低閾値
    CANNY_THRESHOLD1 = 100
    # Cannyエッジ検出の高閾値
    CANNY_THRESHOLD2 = 150
    # ハフ変換の投票数の閾値
    HOUGH_THRESHOLD = 35
    # 検出する線の最小長
    HOUGH_MIN_LINE_LENGTH = 35
    # 線上の点と見なすための最大間隔
    HOUGH_MAX_LINE_GAP = 10
    # ノイズとみなす最小面積（ピクセル数）▼▼▼
    MIN_NOISE_AREA = 45
    
    #リサイズとグレースケール変換
    resized_frame = cv2.resize(color_image, (RESIZE_WIDTH, resize_h), interpolation=cv2.INTER_AREA)
    height, width = resized_frame.shape[:2]
    center_x = width // 2
    
    #前処理
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    adjusted = clahe.apply(gray)
    blurred_again = cv2.GaussianBlur(adjusted, (7,7), 0)
    
    #Cannyでエッジ検出
    edges = cv2.Canny(blurred_again, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)

    # ノイズ除去後のエッジ画像を格納するための黒い画像を用意
    cleaned_edges = np.zeros_like(edges)

    # 各連結成分をループ処理（ラベル0は背景なので1から始める）
    for i in range(1, num_labels):
        # 面積を取得
        area = stats[i, cv2.CC_STAT_AREA]
        # 面積が閾値より大きい場合のみ、その成分を新しい画像に描画
        if area > MIN_NOISE_AREA:
            cleaned_edges[labels == i] = 255
            
    #確率的ハフ変換
    lines = cv2.HoughLinesP(cleaned_edges, 1, np.pi/180, threshold=HOUGH_THRESHOLD, minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)
    
    # 描画用のカラー画像を作成
    line_image = np.copy(resized_frame)
    
    diagonal_lines = []
     #消失点の計算
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            abs_angle_deg = abs(angle_deg)
            
            is_horizontal = (abs_angle_deg <= 10) or (abs_angle_deg >= 175)
            is_vertical = (50 <= abs_angle_deg <= 130)

            if is_horizontal or is_vertical:
                continue

            if x1 == x2: # 垂直線の場合
                continue
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
            
            diagonal_lines.append((m, c))
            
            # 画面端まで線を延長して描画
            points = []
            if m != 0:
                x_at_y0 = -c / m
                if 0 <= x_at_y0 <= width: points.append((int(x_at_y0), 0))
                x_at_y_height = (height - c) / m
                if 0 <= x_at_y_height <= width: points.append((int(x_at_y_height), height))
            y_at_x0 = c
            if 0 <= y_at_x0 <= height: points.append((0, int(y_at_x0)))
            y_at_x_width = m * width + c
            if 0 <= y_at_x_width <= height: points.append((width, int(y_at_x_width)))
        
            if len(points) >= 2:
                cv2.line(line_image, points[0], points[1], (0, 255, 0), 2)
            
    
    intersection_points = []
    if len(diagonal_lines) >= 2:
        for i in range(len(diagonal_lines)):
            for j in range(i + 1, len(diagonal_lines)):
                m1, c1 = diagonal_lines[i]
                m2, c2 = diagonal_lines[j]

                if abs(m1 - m2) < 1e-5: 
                    continue
                
                if m1*m2>0:
                    continue
                
                x = (c2 - c1) / (m1 - m2)
                y = m1 * x + c1

                if -width < x < width * 2 and -height < y < height * 2:
                    intersection_points.append((x, y))
    
    steering_command = "F\n"
    
    if intersection_points:
        x_coords = [p[0] for p in intersection_points]
        vp_x = int(np.median(x_coords))
        cv2.circle(line_image, (vp_x, height//2), 10, (0, 0, 255), -1)
        
        if 0 <= vp_x < width:
            missing_difference = vp_x - center_x 
            if abs(missing_difference) > STEERING_THRESHOLD:
                if missing_difference > 0:
                    steering_command = f"R {missing_difference:.2f}\n" 
                else:
                    steering_command = f"L {abs(missing_difference):.2f}\n"
    
    cv2.line(line_image, (center_x, 0), (center_x, height), (0, 255, 0), 1)
    #return steering_command
    return steering_command, line_image

#検査範囲を決めて結果を返す(障害物のmain)
def detect_step_or_obstacle(depth_img, color_img, camera):
    if depth_img is None:
        return "ERROR", color_img

    h, w = depth_img.shape
    
    # --- 1. スキャン領域（レーン）の設定 ---
    # 画面の幅に対して、左・中・右の領域を定義（幅は全体の10%程度）
    lane_width = 60 # 幅の15%
    center_x = w // 2
    
    # 左、中央、右のX座標範囲 (start_x, end_x)
    lanes = {
        "LEFT":   ( center_x//2 - lane_width//2, center_x//2 + lane_width//2),       # 左側
        "CENTER": (center_x - lane_width//2, center_x + lane_width//2), # 中央
        "RIGHT":  ( (center_x//2)*3 - lane_width//2, (center_x//2)*3 + lane_width//2)        # 右側
    }
    
    # 結果格納用
    lane_status = {} # Example: {"LEFT": True, "CENTER": False, ...} True=Blocked
    
    # --- 2. 各レーンの解析 ---
    for name, (sx, ex) in lanes.items():
        # 下部で定義したヘルパー関数を使用
        is_blocked, blocked_y = analyze_vertical_strip(depth_img, sx, ex, camera)
        lane_status[name] = is_blocked
        
        # --- 描画処理 ---
        color = (0, 255, 0) # 緑: クリア
        if is_blocked:
            color = (0, 0, 255) # 赤: 障害物あり
            # 障害物位置にラインを引く
            cv2.line(color_img, (sx, blocked_y), (ex, blocked_y), (0, 255, 255), 2)
            
        # 枠の描画
        cv2.rectangle(color_img, (sx, 0), (ex, h), color, 1)
        cv2.putText(color_img, name[0], (sx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    scan_top = (h // 2) - 20
    scan_bottom = h - 40
    
    # 上限ライン (ピンク色)
    cv2.line(color_img, (0, scan_top), (w, scan_top), (255, 0, 255), 1)
    # 下限ライン (ピンク色)
    cv2.line(color_img, (0, scan_bottom), (w, scan_bottom), (255, 0, 255), 1)
    
    # --- 3. 総合判定ロジック ---
    # L=Blocked, C=Blocked, R=Blocked -> 段差 (STEP)
    # L=Clear,   C=Blocked, R=Clear   -> 障害物 (OBSTACLE)
    # その他 -> 状況に応じて (今回はシンプルに判定)

    result_type = "CLEAR"
    
    # 3箇所とも塞がっている -> 「段差」または「壁」
    if lane_status["LEFT"] and lane_status["CENTER"] and lane_status["RIGHT"]:
        result_type = "STEP"
        cv2.putText(color_img, "!!! STEP DETECTED !!!", (w//2 - 150, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return result_type, color_img
    # 中央が塞がっていて、左右どちらかは空いている -> 「障害物」
    elif lane_status["CENTER"] and lane_status["LEFT"]:
        result_type = "OBS_LEFT_CENTER"
        cv2.putText(color_img, "OBSTACLE DETECTED", (w//2 - 100, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        return result_type, color_img
    elif lane_status["CENTER"] and lane_status["RIGHT"]:
        result_type = "OBS_RIGHT_CENTER"
        cv2.putText(color_img, "OBSTACLE DETECTED", (w//2 - 100, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        return result_type, color_img
    elif lane_status["CENTER"]:
        result_type = "OBS_CENTER"
        cv2.putText(color_img, "OBSTACLE DETECTED", (w//2 - 100, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        return result_type, color_img
    elif lane_status["LEFT"]:
        result_type = "OBS_LEFT"
        cv2.putText(color_img, "OBSTACLE DETECTED", (w//2 - 100, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        return result_type,color_img
    elif lane_status["RIGHT"]:
        result_type = "OBS_RIGHT"
        cv2.putText(color_img, "OBSTACLE DETECTED", (w//2 - 100, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        return result_type,color_img
    # 左右の壁に近づきすぎた場合 (オプション)
    #elif lane_status["LEFT"] or lane_status["RIGHT"]:
        # ここでは障害物扱いにはしないが、警告を出すなどの処理が可能
    #    pass

    return "SAFE", color_img

#障害物の処理関数
def analyze_vertical_strip(depth_img, x_start, x_end, camera):
    """
    高速化版: NumPyのベクトル演算を用いてforループを排除
    """
    # 1. 切り出しと平均化 (ここは同じ)
    strip = depth_img[:, x_start:x_end]
    h, _ = strip.shape
    row_means = np.mean(strip, axis=1) * camera.depth_scale
    
    # 2. 1/z 変換と平滑化 (ここは同じ)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_means[row_means < 0.1] = np.nan
        inv_z = 1.0 / row_means
        inv_z = np.nan_to_num(inv_z, nan=0.0, posinf=0.0)

    kernel_size = 15
    kernel = np.ones(kernel_size) / kernel_size
    inv_z_smooth = np.convolve(inv_z, kernel, mode='same')
    
    # --- ここから高速化ロジック ---
    
    # 探索範囲の設定
    scan_bottom = h - 40 # 足元 (インデックス大)
    scan_top = (h // 2) - 20    # 頭上 (インデックス小)
    
    # 安全策: 範囲がおかしい場合は即リターン
    if scan_bottom <= scan_top:
        return False, -1

    # 3. 勾配の一括計算 (Shift & Subtract)
    # ループ内でやっていた diff = smooth[y+5] - smooth[y] を配列全体で行う
    # 配列を5つずらして差分をとる
    # shift_arr[i] は original[i+5] に相当
    shift_step = 5
    shifted_smooth = inv_z_smooth[shift_step:] 
    original_smooth = inv_z_smooth[:-shift_step]
    
    # 長さを合わせるために切り詰めた配列同士で引き算
    diffs = shifted_smooth - original_smooth
    
    # 判定に使用する距離データも長さを合わせる
    vals = original_smooth
    
    ref_idx_start = max(0, h - 40)
    ref_idx_end = max(0, h - 20)
    
    # 該当範囲の勾配と値を取得
    ref_diffs = diffs[ref_idx_start:ref_idx_end]
    ref_vals = vals[ref_idx_start:ref_idx_end]
    
    # 有効なデータ (val > 0.5 つまり 2m以内) のみを抽出するためのマスク
    valid_ref_mask = ref_vals > 0.5
    
    # マスクを適用して有効な勾配だけを取り出す
    valid_slopes = ref_diffs[valid_ref_mask]
    
    # 平均を計算 (データがない場合はデフォルト値 0.05)
    if valid_slopes.size > 0:
        ground_slope = np.mean(valid_slopes)
    else:
        ground_slope = 0.05
    
    # 4. 条件判定 (ブールマスクの作成)
    # 条件: (距離 < 2m) かつ (勾配が平坦)
    # vals > 0.5 は 1/z > 0.5 なので z < 2.0m
    # diffs < 0.01 は壁判定
    # これにより、True/False の配列が一瞬で作られる
    #wall_mask = (vals > 0.5) & (diffs < 0.01)
    limit_dist_m = 1.5
    dist_threshold = 1.0 / limit_dist_m
    wall_mask = (vals > dist_threshold) & (diffs < ground_slope * 0.3)
    # 5. 指定されたY座標範囲 (ROI) だけを切り出す
    # 画像座標は上が0なので、スライスは [scan_top : scan_bottom]
    # ただし、diff計算でサイズが5減っているのでインデックスに注意が必要だが、
    # ざっくり scan_top から scan_bottom の範囲を見れば実用上問題ない
    # 正確には配列の末尾が切れているので、minをとって調整
    roi_start = scan_top
    roi_end = min(scan_bottom, len(wall_mask))
    
    roi_mask = wall_mask[roi_start:roi_end]
    
    # 6. 「連続性」のチェック (Convolutionを使用)
    # "True" が REQUIRED_PIXELS (15個) 連続している場所を探す
    # Trueを1、Falseを0として、サイズ15のカーネルで畳み込みを行う
    REQUIRED_PIXELS = 20
    check_kernel = np.ones(REQUIRED_PIXELS)
    
    # 畳み込み計算。結果が 15.0 になる場所 = 15個連続でTrueだった場所
    convolved = np.convolve(roi_mask.astype(int), check_kernel, mode='valid')
    
    # 15.0 (以上) になっているインデックスがあるか？
    # np.where は条件を満たすインデックスを返す
    hit_indices = np.where(convolved >= REQUIRED_PIXELS - 0.1)[0]
    
    is_blocked = False
    blocked_y = -1
    
    if len(hit_indices) > 0:
        is_blocked = True
        # ヒットした場所の中で、一番「下（手前）」にあるものを採用したい場合
        # スキャンループは下から上だったので、インデックスが大きい方が「下」
        # roi_mask内でのインデックス + カーネルサイズ補正 + roiの開始位置
        
        # 最も手前（足元に近い＝インデックスが大きい）の検出位置を取得
        # convolveの 'valid' モードは配列が縮むので位置補正が必要
        # hit_indices[-1] が一番下の検出位置の始点
        
        best_idx = hit_indices[-1] 
        
        # 座標を元の画像座標に戻す
        # best_idx は ROI内の位置。それにROI開始位置などを足す
        blocked_y = roi_start + best_idx + REQUIRED_PIXELS
        
    return is_blocked, blocked_y

#壁との距離測定を行う関数
def process_wall_distance(depth_image,camera,roi_y1,roi_y2,roi_w,depth_h,depth_w,side):
    
    if side =='right':
        roi_x1 = depth_w - roi_w
        roi_x2 = depth_w
    elif side == 'left':
        roi_x1 = 0
        roi_x2 = roi_w
    else:
        return 0.0
    
    # ROI抽出と計算
    if roi_y1 < 0 or roi_y2 > depth_h or roi_x1 < 0 or roi_x2 > depth_w:
        return 0.0 # 範囲外安全策

    depth_roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
    non_zero_depth = depth_roi[depth_roi > 0] # 0 (測定不能) を除外

    if non_zero_depth.size > 0:
        return np.mean(non_zero_depth) * camera.depth_scale
    else:
        return 0.0
    
    
#距離を入力してコマンドを作成する関数
def calc_avoidance_command(distance,target_wall_side):
    # 壁追従コマンド計算のパラメータ
    TARGET_DISTANCE = 0.8  # 目標距離 (m)
    ERROR_THRESHOLD = 0.1   # 許容誤差 (m)
        
    if distance == 0.0:
        return "F\n"  # 停止コマンド（距離不明）
    elif distance is None:
        return "F\n"  # 停止コマンド（距離不明）
    else:
        error = distance - TARGET_DISTANCE
        if abs(error) < ERROR_THRESHOLD:
            command = "F\n" 
            print(f" [CONTROL] OK ({distance:.2f}m) -> 'S'")
        elif error > 0: # 遠い
            # 右壁ターゲットで遠い(右に寄りたい) -> Right
            # 左壁ターゲットで遠い(左に寄りたい) -> Left
            direction = "Rw" if target_wall_side == 'right' else "Lw"
            command = f"{direction} {abs(error):.2f}\n"
            print(f" [CONTROL] 遠い ({distance:.2f}m) -> '{direction}'")
        else: # 近い (error < 0)
            # 右壁ターゲットで近い(左に避けたい) -> Left
            # 左壁ターゲットで近い(右に避けたい) -> Right
            direction = "Lw" if target_wall_side == 'right' else "Rw"
            command = f"{direction} {abs(error):.2f}\n"
            print(f" [CONTROL] 近い ({distance:.2f}m) -> '{direction}'")
            
    return command