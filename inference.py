import torch
import cv2
import os
import pandas as pd
import ast
from tqdm import tqdm
import warnings
import numpy as np
from datetime import datetime 
warnings.filterwarnings("ignore")

def LoadModel(model_path):
    '''
    Load YOLOv5 model from the specified path.
    
    Args:
        model_path (str): Path to the model file.
        
    Returns:
        YOLOv5 model.
    '''
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model

def FrameInference(image, model, confidence_threshold):
    '''
    Perform inference on a single frame using the YOLOv5 model.
    
    Args:
        image (array): Image frame to perform inference on.
        model: YOLOv5 model object.
        confidence_threshold (float): Confidence threshold for detections.
        
    Returns:
        Inference results.
    '''
    model.conf = confidence_threshold
    inference_results = model(image)
    return inference_results

def GetPredictionDict(frame_count, prediction_list):
    '''
    Create a prediction dictionary based on frame count and prediction list.
    
    Args:
        frame_count (int): Frame number.
        prediction_list (list): List of predictions.
        
    Returns:
        dict: Prediction dictionary.
    '''
    prediction_dict = {}
    prediction_dict['frame_number'] = frame_count
    prediction_dict['predictions'] = prediction_list
    prediction_dict['prediction_found'] = 0
    prediction_dict['AA_predicted'] = 0
    prediction_dict['chokeslam_predicted'] = 0
        
    if len(prediction_list) > 0:
        prediction_dict['prediction_found'] = 1
        
        for prediction in prediction_list:
            if prediction['name'] == 'chokeslam':
                prediction_dict['chokeslam_predicted'] = 1
            elif prediction['name'] == 'AA':
                prediction_dict['AA_predicted'] = 1
    return prediction_dict

def RunInference(model_path, video_path, output_path, frame_interval=10, confidence_threshold=0.5):
    '''
    Run inference on a video and save inferred video. Also returns DataFrame with framewise predictions.
    
    Args:
        model_path (str): Path to the YOLOv5 model file.
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video.
        frame_interval (int): Interval between processed frames. Default is 10.
        confidence_threshold (float): Confidence threshold for detections. Default is 0.5.
        
    Returns:
        pd.DataFrame: DataFrame with framewise predictions.
    '''
    frame_count = 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = LoadModel(model_path)
    Prediction_df = pd.DataFrame(columns=['frame_number', 'predictions', 'prediction_found', 'AA_predicted', 'chokeslam_predicted'])
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    datetime_now = str(datetime.now().strftime('%d%m_%H%M'))
    video_name = f'inf_{datetime_now}.avi'
    inf_video = cv2.VideoWriter(os.path.join(output_path, video_name), cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    pbar = tqdm(desc='progress', total=total_frames)
    temp_list = []
    while cap.isOpened() and frame_count < total_frames:
        _, frame = cap.read()
        if frame_count % frame_interval == 0:
            try:
                inference_results = FrameInference(frame, model, confidence_threshold)
                prediction_list = inference_results.pandas().xyxy[0].to_json(orient="records")
                prediction_list = ast.literal_eval(prediction_list)
                prediction_dict = GetPredictionDict(frame_count, prediction_list)
                inf_video.write(np.squeeze(inference_results.render()))
            except Exception as e:
                print(e)
                break
            
            temp_list.append(prediction_dict)
        
        frame_count += 1
        pbar.update(1)

    Prediction_df = pd.DataFrame(temp_list)
    pbar.close()
    print(f'Inference Video saved at {output_path}{video_name}')
    return Prediction_df
