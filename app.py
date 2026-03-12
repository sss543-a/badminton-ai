import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tempfile
import os

# ======== MediaPipe 初始化 (增加錯誤捕捉) ========
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError as e:
    st.error("MediaPipe 模組載入失敗，請檢查 packages.txt 是否包含 libgl1-mesa-glx")
    st.stop()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# ======== Streamlit 網頁介面 ========
st.set_page_config(page_title="羽球姿勢分析", page_icon="🏸")
st.title("羽球動作技術AI 分析系統 öㅅö")

if 'analyzed_path' not in st.session_state:
    st.session_state.analyzed_path = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

uploaded_file = st.file_uploader("選擇影片檔案...", type=["mp4", "mov", "avi"])

if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_file:
    st.session_state.analyzed_path = None
    st.session_state.last_uploaded_file = uploaded_file.name

if uploaded_file is not None:
    if st.session_state.analyzed_path is None:
        # 使用更安全的方式處理暫存檔
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            input_path = tfile.name
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = os.path.join(tempfile.gettempdir(), "analyzed_video.mp4")
        # 注意：在 Linux/Streamlit Cloud 上，'avc1' (H.264) 通常比 'mp4v' 更好下載播放
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (1080, 640))

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.resize(frame, (1080, 640))
                # 轉換顏色空間
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    h, w = frame.shape[:2]
                    
                    # 抓取右側關節 (確保 Landmark 存在)
                    try:
                        shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
                        elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h]
                        wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h]

                        angle = calculate_angle(shoulder, elbow, wrist)
                        cv2.putText(frame, f"Angle: {int(angle)}", (30, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    except:
                        pass
                    
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                out.write(frame)
                count += 1
                if frame_count > 0:
                    progress_bar.progress(min(count / frame_count, 1.0))
                status_text.text(f"正在分析... {count}/{frame_count} 幀")

        cap.release()
        out.release()
        
        st.session_state.analyzed_path = output_path
        st.rerun()

    else:
        st.success("分析完成！可以點擊下方按鈕下載囉 (σ′▽‵)′▽‵)σ")
        with open(st.session_state.analyzed_path, "rb") as file:
            st.download_button(
                label="下載分析結果影片",
                data=file,
                file_name="badminton_analysis.mp4",
                mime="video/mp4"
            )
        
        if st.button("重新分析該影片"):
            st.session_state.analyzed_path = None
            st.rerun()
