import streamlit as st
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import os
import tempfile
from pathlib import Path
import imageio
from moviepy.editor import VideoFileClip
import time

class GIFBackgroundRemover:
    def __init__(self):
        """تهيئة المعالج مع إعدادات محسنة"""
        self.output_dir = Path(tempfile.gettempdir()) / 'gif_processing'
        self.output_dir.mkdir(exist_ok=True)
        
    def video_to_gif(self, video_path, fps=15):  # زيادة معدل الإطارات للحصول على حركة أكثر سلاسة
        """تحويل الفيديو إلى GIF مع تحسين الجودة"""
        clip = VideoFileClip(video_path)
        # تحسين جودة التحويل
        gif_path = str(self.output_dir / f'temp_{int(time.time())}.gif')
        clip.write_gif(
            gif_path,
            fps=fps,
            program='ffmpeg',  # استخدام ffmpeg للحصول على جودة أفضل
            opt='optimal'  # استخدام الإعدادات المثلى
        )
        clip.close()
        return gif_path
        
    def process_gif(self, gif_path, progress_bar, status_text):
        """معالجة الـ GIF مع تحسين الشفافية والجودة"""
        gif = Image.open(gif_path)
        frames = []
        n_frames = gif.n_frames
        
        for i in range(n_frames):
            gif.seek(i)
            # تحويل الإطار إلى RGBA للحفاظ على الشفافية
            frame = gif.convert('RGBA')
            
            # تحسين إزالة الخلفية مع إعدادات إضافية
            frame_no_bg = remove(
                frame,
                alpha_matting=True,  # تمكين alpha matting للحصول على حواف أفضل
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=5
            )
            
            # تحويل الخلفية السوداء إلى شفافة
            frame_data = np.array(frame_no_bg)
            r, g, b, a = frame_data.T
            black_areas = (r == 0) & (g == 0) & (b == 0)
            frame_data[..., 3][black_areas.T] = 0
            
            # تحويل مصفوفة NumPy إلى صورة PIL
            frame_processed = Image.fromarray(frame_data)
            frames.append(frame_processed)
            
            progress = (i + 1) / n_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame: {i+1}/{n_frames}")
        
        # حفظ النتيجة مع إعدادات محسنة للجودة
        output_path = str(self.output_dir / f'output_{int(time.time())}.gif')
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000//15,  # تحسين توقيت الإطارات
            loop=0,
            optimize=False,  # تعطيل التحسين للحفاظ على الجودة
            quality=95,  # جودة عالية
            disposal=2  # تحسين معالجة الإطارات المتتالية
        )
        
        return output_path

def validate_video(file):
    """التحقق من صحة الفيديو"""
    if file is None:
        return False, "No file uploaded"
        
    file_size = len(file.getvalue()) / (1024 * 1024)
    if file_size > 100:
        return False, f"File too large: {file_size:.1f}MB (max 100MB)"
        
    return True, "File is valid"

def main():
    st.set_page_config(page_title="Video to GIF Background Remover", layout="wide")
    
    st.title("🎥 Video to GIF Background Remover")
    st.write("Upload a video to convert it to GIF and remove its background. Limited to 100MB.")
    
    # إضافة خيارات متقدمة
    with st.expander("Advanced Settings"):
        fps = st.slider("FPS (Frames Per Second)", min_value=10, max_value=30, value=15)
        quality = st.slider("Output Quality", min_value=70, max_value=100, value=95)
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file:
        is_valid, message = validate_video(uploaded_file)
        
        if not is_valid:
            st.error(message)
            return
            
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.read())
        temp_input.close()
        
        try:
            if st.button("Remove Background"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                processor = GIFBackgroundRemover()
                
                status_text.text("Converting to GIF...")
                gif_path = processor.video_to_gif(temp_input.name, fps=fps)
                
                output_path = processor.process_gif(gif_path, progress_bar, status_text)
                
                st.success("Processing complete!")
                st.image(output_path)
                
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download GIF",
                        data=file,
                        file_name="video_no_background.gif",
                        mime="image/gif"
                    )
                
        finally:
            os.unlink(temp_input.name)
            
    st.markdown("---")
    st.markdown("""
        ### Tips for best results:
        1. Use videos with clear subjects and contrasting backgrounds
        2. Keep the video short (5-10 seconds recommended)
        3. Ensure good lighting in the video
        4. Avoid very fast movements
        5. Use higher FPS for smoother animation
    """)

if __name__ == "__main__":
    main()