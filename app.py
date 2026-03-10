import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from collections import deque

# ======== 1. 初始化 MediaPipe & 介面設定 ========
st.set_page_config(page_title="🏸 AI 羽球分析", layout="wide")
st.title("🏸 反手發球姿勢 AI 雲端分析系統")
st.info("請上傳影片，系統將自動分析你的發球穩定度與手肘角度。")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ======== 2. 輔助函數 (邏輯核心) ========
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def calculate_speed(prev, curr, fps):
    if prev is None or curr is None: return 0
    dist = np.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
    return dist * fps

# ======== 3. 檔案上傳與處理 ========
uploaded_file = st.file_uploader("選擇你的發球影片 (mp4/mov)", type=["mp4", "mov"])

if uploaded_file:
    # 建立暫存檔讀取影片
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 用於 Streamlit 即時顯示的容器
    frame_placeholder = st.empty()
    
    # 軌跡與歷史紀錄
    trajectory = deque(maxlen=30)
    angle_history = deque(maxlen=30)
    prev_wrist = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 影像前處理
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # 這裡使用右手作為例子 (可根據需求改為左手)
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
                elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height]
                wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height]

                # A. 計算角度與顏色警告
                angle = calculate_angle(shoulder, elbow, wrist)
                angle_history.append(angle)
                color = (0, 255, 0) # 綠色 (正常)
                msg = f"Angle: {int(angle)} deg"
                if angle < 100: color, msg = (255, 0, 0), "Too Bent!"
                elif angle > 165: color, msg = (255, 165, 0), "Too Straight!"

                # B. 計算穩定度 (標準差反轉)
                consistency = max(0, 100 - np.std(angle_history)) if len(angle_history) > 5 else 100

                # C. 繪製
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Consistency: {int(consistency)}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # D. 軌跡速度變化
                trajectory.append(tuple(map(int, wrist)))
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 255), 3)

            # 在網頁上顯示當前影格
            frame_placeholder.image(frame, channels="BGR")

    cap.release()
    st.success("分析結束！")