import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tempfile
import os

# ======== MediaPipe 初始化 ========
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# ======== Streamlit 網頁介面 ========
st.set_page_config(page_title="羽球姿勢分析", page_icon="🏸")
st.title("羽球動作技術AI 分析系統 öㅅö")

# 初始化 session_state，用來記憶是否已經分析過
if 'analyzed_path' not in st.session_state:
    st.session_state.analyzed_path = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

uploaded_file = st.file_uploader("選擇影片檔案...", type=["mp4", "mov", "avi"])

# 如果使用者換了新檔案，清除舊的分析紀錄
if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_file:
    st.session_state.analyzed_path = None
    st.session_state.last_uploaded_file = uploaded_file.name

if uploaded_file is not None:
    # 情況 A：還沒分析過，開始分析
    if st.session_state.analyzed_path is None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 設定輸出
        output_path = os.path.join(tempfile.gettempdir(), "analyzed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1080, 640))

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        angle_history = deque(maxlen=30)
        trajectory = deque(maxlen=30)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.resize(frame, (1080, 640))
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    h, w = frame.shape[:2]
                    
                    shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
                    elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h]
                    wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h]

                    angle = calculate_angle(shoulder, elbow, wrist)
                    angle_history.append(angle)
                    
                    # 繪製邏輯 (維持你原本的邏輯)
                    cv2.putText(frame, f"Angle: {int(angle)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                out.write(frame)
                count += 1
                progress_bar.progress(count / frame_count)
                status_text.text(f"正在分析... {int(count/frame_count*100)}%")

        cap.release()
        out.release()
        
        # 核心：分析完後把路徑存進 session_state
        st.session_state.analyzed_path = output_path
        st.rerun() # 強制刷新以顯示下載按鈕

    # 情況 B：已經分析好了，直接顯示下載按鈕
    else:
        st.success("分析完成！可以點擊下方按鈕下載囉 (σ′▽‵)′▽‵)σ")
        with open(st.session_state.analyzed_path, "rb") as file:
            st.download_button(
                label="下載分析結果影片",
                data=file,
                file_name="badminton_analysis.mp4",
                mime="video/mp4"
            )
        
        # 提供一個按鈕讓使用者可以清除紀錄重新分析
        if st.button("重新分析該影片"):
            st.session_state.analyzed_path = None

            st.rerun()
