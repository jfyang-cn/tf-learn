import os, datetime, sys, argparse
import cv2
import numpy as np
import json
import time
import tensorflow as tf
from builder import ModelBuilder

print(tf.__version__)

if tf.__version__ == '1.14.0':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    tf_config = ConfigProto()
    tf_config.gpu_options.allow_growth = True

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def webcam(config, weight_path, stream_path, output_path):
    
    input_width        = config['model']['input_width']
    input_height       = config['model']['input_height']
    input_depth        = config['model']['input_depth']
    label_file         = config['model']['labels']
    model_name         = config['model']['name']
    
    train_data_dir     = config['train']['data_dir']
    train_file_list    = config['train']['file_list']
    pretrained_weights = config['train']['pretrained_weights']
    batch_size         = config['train']['batch_size']
    learning_rate      = config['train']['learning_rate']
    nb_epochs          = config['train']['nb_epochs']
    start_epoch        = config['train']['start_epoch']
    train_base         = config['train']['train_base']
    
    valid_data_dir     = config['valid']['data_dir']
    valid_file_list    = config['valid']['file_list']

    builder = ModelBuilder(config)
    
    # train
    graph = tf.Graph()
    sess = tf.Session(graph=graph,config=tf_config)

    tf.keras.backend.set_session(sess)
    with graph.as_default():
        model = builder.build_model()
        model.load_weights(weight_path)
    
    ### Define empty sliding window
    frame_window = np.empty((0, input_width, input_height, 3)) # seq, dim0, dim1, channel

    ### State Machine Define
    RUN_STATE = 0
    WAIT_STATE = 1
    SET_NEW_ACTION_STATE = 2
    state = RUN_STATE # 
    previous_action = -1 # no action
    text_show = 'no action'

    # Class label define
    class_text = ['debris','rockfail','rain']
#     class_text = [
#     '1 Horizontal arm wave',
#     '2 High arm wave',
#     '3 Two hand wave',
#     '4 Catch Cap',
#     '5 High throw',
#     '6 Draw X',
#     '7 Draw Tick',
#     '8 Toss Paper',
#     '9 Forward Kick',
#     '10 Side Kick',
#     '11 Take Umbrella',
#     '12 Bend',
#     '13 Hand Clap',
#     '14 Walk',
#     '15 Phone Call',
#     '16 Drink',
#     '17 Sit down',
#     '18 Stand up']

    # class_text = [
    # '1 Horizontal arm wave',
    # '2 High arm wave',
    # '3 Two hand wave',
    # '4 Catch Cap',
    # # '5 High throw',
    # # '6 Draw X',
    # # '7 Draw Tick',
    # # '8 Toss Paper',
    # # '9 Forward Kick',
    # # '10 Side Kick',
    # # '11 Take Umbrella',
    # # '12 Bend',
    # '13 Hand Clap',
    # '14 Walk',
    # # '15 Phone Call',
    # # '16 Drink',
    # # '17 Sit down',
    # # '18 Stand up'
    # ]
    
    cap = cv2.VideoCapture(stream_path)

    if output_path is not None:
        sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = 25
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vout = cv2.VideoWriter()
#         vout.open('output.mp4',fourcc,fps,(768,576),True)    # 4:3
#         vout.open(output_path,fourcc,fps,(800,800),True)
        vout.open(output_path,fourcc,fps,sz,True)

    start_time = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()  

        if ret == True:
#             frame = cv2.resize(frame, (1024,576))
#             frame = frame[0:576,128:128+768]         # 4:3
#             frame = frame[100:900,400:1200]

            new_f = cv2.resize(frame, (input_width, input_height))
            new_f_rs = np.reshape(new_f, (1, *new_f.shape))
            frame_window = np.append(frame_window, new_f_rs, axis=0)            

            ### if sliding window is full(8 frames), start action recognition
            if frame_window.shape[0] >= input_depth:
                
                ### Predict action from model
                input_0 = frame_window.reshape(1, *frame_window.shape)
                with graph.as_default():
                    output = model.predict(input_0)[0]
                predict_ind = np.argmax(output)

                ### Check noise of action
                if output[predict_ind] < 0.70:
                    new_action = -1 # no action(noise)
                else:
                    new_action = predict_ind # action detect

                ### Use State Machine to delete noise between action(just for stability)
                ### RUN_STATE: normal state, change to wait state when action is changed
                if state == RUN_STATE:
                    if new_action != previous_action: # action change
                        state = WAIT_STATE
                        start_time = time.time()     
                    else:
                        if previous_action == -1:# or previous_action == 5:
                            text_show = 'no action'                                              
                        else:
                            text_show = "{: <10} {:.2f} ".format(class_text[previous_action],
                                        output[previous_action] )
                        print(text_show)  

                ### WAIT_STATE: wait 0.5 second when action from prediction is change to fillout noise
                elif state == WAIT_STATE:
                    dif_time = time.time() - start_time
                    if dif_time > 0.5: # wait 0.5 second
                        state = RUN_STATE
                        previous_action = new_action

                ### put text to image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text_show, (10,50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)   

                ### shift sliding window
                frame_window = frame_window[1:input_depth]

                if output_path is not None:
                    vout.write(frame)
                else:
                    ## To show dif RGB image
                    vis = np.concatenate((new_f, frame_window_new[0,n_sequence-1]), axis=0)
                    cv2.imshow('Frame', vis)
                    cv2.imshow('Frame', frame)                    


            ### To show FPS
            # end_time = time.time()
            # diff_time =end_time - start_time
            # print("FPS:",1/diff_time)
            # start_time = end_time

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break 
        else: 
            break

    vout.release()
    cap.release()


def main(args):
    
    config_path = args.conf
    stream_path = args.stream
    weight_path = args.weight
    output_path = args.output
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())    
        webcam(config, weight_path, stream_path, output_path)

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
    
    parser.add_argument(
        '-s',
        '--stream',
        help='stream url e.g. rtsp://192.168.1.1/0')
    
    parser.add_argument(
        '-w',
        '--weight',
        help='model weight path')
    
    parser.add_argument(
        '-o',
        '--output',
        help='output file name',
        default=None)
        
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
