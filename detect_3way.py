import cv2
import numpy as np
import math

# コマンド送信用の関数
def send_motor_command(ser, command):
    ser.write(command.encode('utf-8'))
    ser.flush()
    print(f"Sending command: {command}")
    
    
def detect_step_or_obstacle(depth_img, color_img, camera):
    """
    3本の縦スキャンを行い、段差(全幅)か障害物(局所)かを判定する関数
    
    Returns:
        result_type (str): "CLEAR", "OBSTACLE", "STEP" のいずれか
        result_img (img): 可視化結果
    """
    if depth_img is None:
        return "ERROR", color_img

    h, w = depth_img.shape
    
    # --- 1. スキャン領域（レーン）の設定 ---
    # 画面の幅に対して、左・中・右の領域を定義（幅は全体の10%程度）
    lane_width = 40 # 幅の15%
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

    # 中央が塞がっていて、左右どちらかは空いている -> 「障害物」
    elif lane_status["CENTER"]:
        result_type = "OBSTACLE"
        
        # 回避方向の提案（デバッグ表示）
        """
        avoid_msg = "AVOID"
        if not lane_status["LEFT"]:
            avoid_msg = "<< AVOID LEFT"
        elif not lane_status["RIGHT"]:
            avoid_msg = "AVOID RIGHT >>"
        """ 
        cv2.putText(color_img, "OBSTACLE DETECTED", (w//2 - 100, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

    # 左右の壁に近づきすぎた場合 (オプション)
    #elif lane_status["LEFT"] or lane_status["RIGHT"]:
        # ここでは障害物扱いにはしないが、警告を出すなどの処理が可能
    #    pass

    return result_type, color_img


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
    wall_mask = (vals > 0.5) & (diffs < ground_slope * 0.3)
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
    REQUIRED_PIXELS = 15
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