import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
from PIL import Image

# تحسين الواجهة باستخدام CSS مخصص
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stFileUploader {
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# عنوان التطبيق
st.title("🎥 Remove Background from Video and Replace It")

# تحميل الفيديو
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # حفظ الفيديو المؤقت
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    # خيارات للمستخدم
    col1, col2 = st.columns(2)
    with col1:
        option = st.radio("Choose an option:", ["Remove Background Only", "Replace Background with Image"])
    with col2:
        start_processing = st.button("Start Processing")

    # تحميل صورة الخلفية (فقط إذا تم اختيار استبدال الخلفية)
    if option == "Replace Background with Image":
        background_image = st.file_uploader("Upload a background image", type=["jpg", "png", "jpeg"])
    else:
        background_image = None

    if start_processing:
        # تهيئة نموذج إزالة الخلفية
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        # قراءة الفيديو
        cap = cv2.VideoCapture("temp_video.mp4")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # مكان لعرض الفيديو
        video_placeholder = st.empty()

        # تحميل صورة الخلفية (إذا تم اختيار استبدال الخلفية)
        if option == "Replace Background with Image" and background_image is not None:
            background_bytes = background_image.read()
            background = cv2.imdecode(np.frombuffer(background_bytes, np.uint8), cv2.IMREAD_COLOR)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)  # تحويل إلى RGB
            background = cv2.resize(background, (width, height))  # تغيير حجم الخلفية لتناسب الفيديو
        else:
            background = None

        # إعداد الفيديو المُنتَج
        output_video_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # شريط التقدم
        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # تحويل الإطار إلى RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # تطبيق نموذج إزالة الخلفية
            results = segmentation.process(frame_rgb)
            mask = results.segmentation_mask

            # تحسين القناع باستخدام Gaussian Blur
            mask = cv2.GaussianBlur(mask, (15, 15), 0)

            # إنشاء القناع
            condition = np.stack((mask,) * 3, axis=-1) > 0.6

            # إذا تم اختيار "Remove Background Only"
            if option == "Remove Background Only":
                output = np.where(condition, frame_rgb, 0)  # إزالة الخلفية (تعيينها إلى اللون الأسود)
            else:
                # إذا تم اختيار "Replace Background with Image"
                if background is not None:
                    # استبدال الخلفية
                    output = np.where(condition, frame_rgb, background)
                else:
                    st.warning("Please upload a background image to replace it.")
                    break

            # عرض الإطار المعالج
            video_placeholder.image(output, channels="RGB", use_container_width=True)

            # حفظ الإطار المعالج في الفيديو المُنتَج
            out.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

            # تحديث شريط التقدم
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {progress}%")

        # إطلاق الفيديو وإغلاق النوافذ
        cap.release()
        out.release()

        # دمج الصوت مع الفيديو المُنتَج
        final_output = "final_output.mp4"
        os.system(f"ffmpeg -i output_video.mp4 -i temp_video.mp4 -c:v copy -map 0:v:0 -map 1:a:0 -shortest {final_output}")

        # زر لتحميل الفيديو المُنتَج
        with open(final_output, "rb") as f:
            video_data = f.read()
        st.download_button(
            label="Download Processed Video",
            data=video_data,
            file_name="final_output.mp4",
            mime="video/mp4",
        )

        # تنظيف الملفات المؤقتة
        os.remove("temp_video.mp4")
        os.remove("output_video.mp4")
        os.remove("final_output.mp4")
else:
    st.warning("Please upload a video to start processing.")


    
# إخفاء العناصر غير المرغوب فيها
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            #stStreamlitLogo {display: none;}
            a {
                text-decoration: none;
                color: inherit;
                pointer-events: none;
            }
            a:hover {
                text-decoration: none;
                color: inherit;
                cursor: default;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)