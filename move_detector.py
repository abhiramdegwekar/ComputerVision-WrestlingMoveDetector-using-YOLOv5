import streamlit as st
import tempfile
from inference import *
from video_utils import *

def app():
    
    st.title("Wrestling Move Detector")

    if docx_file := st.file_uploader("Upload a video file", type=["mp4"]):
        video_bytes = docx_file.read()
        st.video(video_bytes)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_bytes)
        video_path=tfile.name

        x=st.slider("Frame Interval", min_value=1,max_value=30,value=1,step=1)
        y=st.slider("Confidence Threshold", min_value=0.1,max_value=1.0,value=0.8,step=0.025)

    if st.button("Submit"):
        run(video_path, x, y)


def run(video_path, x, y):
    model_path=r"./weights/best.pt"
    output_path=r"./output/"

    prediction=RunInference(model_path,video_path,output_path,frame_interval = x,confidence_threshold = y)

    #AA smoothing filter
    AA_timestamps = operation(prediction['frame_number'],prediction['AA_predicted'],window=8, continuous_positives_window=6,threshold=0.6, debug = False)

    #chokeslam smoothing filter
    chokeslam_timestamps = operation(prediction['frame_number'],
    prediction['chokeslam_predicted'],
    window=8, 
    continuous_positives_window=8,
    threshold=0.8, debug = False)
    st.header("Results:")
    if len(chokeslam_timestamps)>0:
        timestamps_func(
            chokeslam_timestamps,
            'Chokeslam was detected from ',
        )
    elif len(AA_timestamps)>0:
        timestamps_func(
            AA_timestamps,
            'Attitude Adjustment was detected from ',
        )

app()