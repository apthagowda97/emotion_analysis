from input_feeder import InputFeeder
from face_detection import FaceDetection
from emotions_recognition import EmotionsRecognition
from face_reidentification import FaceReidentification

import time
import os
import cv2
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from argparse import ArgumentParser


parent_dir =  os.path.split(os.getcwd())[0]
input_video = os.path.join(parent_dir,'input','netflix.mp4')

emotions = {0:'neutral',1:'happy',2:'sadness',3:'surprise',4:'anger'}
faces_db = {}
emotions_db = {}
face_vectors_db = []

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-t", "--input_type", type=str,default="video",
                        help="'video' or 'cam' input type")
    parser.add_argument("-i", "--input_file",type=str,default=input_video,
                        help="Path to image or video file")

    parser.add_argument("-dp", "--detection_model_precision", type=str,default='FP32-INT8',
                        help="face detection model precision")
    parser.add_argument("-ep", "--emotion_model_precision", type=str,default='FP32-INT8',
                        help="emotion recognition model precision")
    parser.add_argument("-rp", "--reidentification_model_precision", type=str,default='FP32-INT8',
                        help="face reidentification model precision")

    return parser

def update_(face,face_emotion,face_vector):
    
    similarity = [0]
    for face_vector_cmp in face_vectors_db:
        similarity.append(cosine_similarity(face_vector,face_vector_cmp))
    
    if max(similarity)>0.5:
        faces_db[np.argmax(similarity)-1].append(face)
        emotions_db[np.argmax(similarity)-1].append(face_emotion)
        
    else:
        face_vectors_db.append(face_vector)
        faces_db[len(faces_db)] = [face]
        emotions_db[len(emotions_db)] = [face_emotion]

def add_emotion_emoji(face_emotion,cord,bbox_frame):
    l = cord[0]
    m = cord[0]+20
    n = cord[1]+cord[3]-20
    o = cord[1]+cord[3]
    bbox_frame[n:o,l:m,:] = cv2.resize(cv2.imread(os.path.join(parent_dir,'bin',f'{emotions[face_emotion]}.png')),(20,20))

def plot_emotion_count(bbox_frame,emotions_count):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.axis('off')
    ax.set_ylim(0,len(emotions_count))
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    ax.bar(emotions.values(),emotions_count,color='red')
    canvas.draw()
    w, h = (fig.get_size_inches() * fig.get_dpi()).astype(int)
    x = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h,w,3)
    x = cv2.resize(x, (216,144))
    bbox_frame[276:420,504:720,2] = np.where(x[:,:,2]==0,255,bbox_frame[276:420,504:720,2])
    bbox_frame[420:450,534:564,:] = cv2.resize(cv2.imread(os.path.join(parent_dir,'bin','neutral.png')),(30,30))
    bbox_frame[420:450,568:598,:] = cv2.resize(cv2.imread(os.path.join(parent_dir,'bin','happy.png')),(30,30))
    bbox_frame[420:450,602:632,:] = cv2.resize(cv2.imread(os.path.join(parent_dir,'bin','sadness.png')),(30,30))
    bbox_frame[420:450,636:666,:] = cv2.resize(cv2.imread(os.path.join(parent_dir,'bin','surprise.png')),(30,30))
    bbox_frame[420:450,670:700,:] = cv2.resize(cv2.imread(os.path.join(parent_dir,'bin','anger.png')),(30,30))

def inference(args):
    
    time_sheet = {'detection_infr':[],'emotion_infr':[],'reidentification_infr':[],'infr_per_frame':[]}
    
    output_video = os.path.join(parent_dir,'output',os.path.split(args.input_file)[1])
    out = cv2.VideoWriter(output_video, 0x00000021, 18, (720,480))
    
    logging.basicConfig(filename='result.log',level=logging.INFO)
    logging.info("=================================================================================")
    logging.info("Precision(detection,emotions,reidentification): FP{0},FP{1},FP{2}".format(\
            args.detection_model_precision,args.emotion_model_precision,args.reidentification_model_precision))

    model_load_start = time.time()

    face_detection= FaceDetection(precision = args.detection_model_precision)
    face_detection.load_model()
    emotions_recognition = EmotionsRecognition(precision = args.emotion_model_precision)
    emotions_recognition.load_model()
    face_reidentification = FaceReidentification(precision = args.reidentification_model_precision)
    face_reidentification.load_model()
    
    logging.info("3 models load time: {0:.4f}sec".format(time.time()-model_load_start))

    input_feeder = InputFeeder(args.input_type,args.input_file)
    input_feeder.load_data()

    cv2.namedWindow('preview', cv2.WND_PROP_FULLSCREEN)
    
    total_infr_start = time.time()

    for frame in input_feeder.next_batch():
        if frame is None:
            break

        frame =  cv2.resize(frame, (720,480))
        detection_infr_start = time.time()
        bbox_frame,faces,cords = face_detection.predict(frame)
        time_sheet['detection_infr'].append(time.time()-detection_infr_start)

        emotions_count = [0,0,0,0,0]
        for index,face in enumerate(faces):

            emotion_infr_start = time.time()
            face_emotion = emotions_recognition.predict(face)
            time_sheet['emotion_infr'].append(time.time()-emotion_infr_start)

            reidentification_infr_start = time.time()
            face_vector  = face_reidentification.predict(face)
            time_sheet['reidentification_infr'].append(time.time()-reidentification_infr_start)

            update_(face,face_emotion,face_vector)
            
            add_emotion_emoji(face_emotion,cords[index],bbox_frame)
            emotions_count[face_emotion] += 1
        plot_emotion_count(bbox_frame,emotions_count)

    
        time_sheet['infr_per_frame'].append(time.time()-detection_infr_start)
        cv2.imshow('preview',bbox_frame)
        out.write(bbox_frame)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    logging.info("face detection model avg inference per frame: {0:.4f}sec".format(np.mean(time_sheet['detection_infr'])))
    logging.info("emotion recognition model avg inference per frame: {0:.4f}sec".format(np.mean(time_sheet['emotion_infr'])))
    logging.info("face reidentification model avg inference per frame: {0:.4f}sec".format(np.mean(time_sheet['reidentification_infr'])))

    logging.info("3 Model avg inference per frame: {0:.4f}sec".format(np.mean(time_sheet['infr_per_frame'])))
    logging.info("Total inference time: {0:.4f}sec".format(time.time()-total_infr_start))
    logging.info("====================================END==========================================\n")

    out.release()
    input_feeder.close()
    cv2.destroyAllWindows()

def main():
    args = build_argparser().parse_args()
    inference(args)

if __name__ == '__main__':
    main()


