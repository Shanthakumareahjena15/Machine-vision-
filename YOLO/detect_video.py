import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import numpy as np

flags.DEFINE_string('classes', './data/labels/obj_fish.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/videos/cod_trial.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', 'output_cod.avi', './data/video/')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')

steinbutt_count = []
kliesche_count = []
herring_count = []
cod_count = []
full_score = []
#dummy_scores = []
dummy_nums = []
fine_class = []
def main(_argv):
    counta = 0
    count =0
    dorsch_counter = 0
    steinbutt_counter = 0
    kliesche_counter = 0
    herring_counter = 0
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    fps = 0.0
    count = 0
    while True:
        _, img = vid.read()
        img_raw = img
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        #img_raw = img_in
        #img_in = img_in[336:535, 787:1198] #cod first
        #img_in = img_in[365:555, 750:1278] #cod second vid
        # img_in = img_in[387:634, 739:1218] #flat_fish
        #img_in = img_in[344: 513, 766: 1042] #steinbutt
        #img_in = img_in[331:520, 789:1065] #liman
        #img_in = img_in[420: 580, 751: 1226] # cod_anne
        img_in= img_in[365:555, 750:1278] # cod_new
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        if not boxes.any():
            dummy_nums.append(0)
            fine_class.append('no fish')
            cod_count.append(0)
            herring_count.append(0)
            kliesche_count.append(0)
            steinbutt_count.append(0)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        img, stack_predections, length = draw_outputs(img[365:555, 750:1278], (boxes, scores, classes, nums), class_names)
        img_raw[365:555, 750:1278] = img #cod_trial
        cv2.putText(img_raw, "FPS: {:.2f}".format(fps), (0, 60),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        full_score.append(scores)
        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]
        if scores.any():
            
            for i in range(nums):
                if (scores[i]*100) > 50:  
                    cv2.putText(img_raw, 'Computed_length = {} cm'.format(round((length[-1]))), (0, 240), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 3)     
                    fine_class.append(stack_predections[0])             
                    if stack_predections[0] ==0:
                        cv2.putText(img_raw, 'Detected Fish = Cod', (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 3)
                        if np.all (np.array(cod_count[-8:]) == 0) and np.all(np.array(herring_count[-8:]) == 0)  and np.all(np.array(kliesche_count[-8:]) == 0) and np.all(np.array(steinbutt_count[-8:]) == 0):
                            dorsch_counter+=1
                            cod_count.append(dorsch_counter)
                        cod_count.append(dorsch_counter)
                    
                    if stack_predections[0] ==1:
                       cv2.putText(img_raw, 'Detected Fish = Herring', (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 3)
                       if np.all (np.array(cod_count[-9:]) == 0) and np.all(np.array(herring_count[-9:]) == 0)  and np.all(np.array(kliesche_count[-9:]) == 0) and np.all(np.array(steinbutt_count[-9:]) == 0):
                           herring_counter+=1
                           herring_count.append(herring_counter)
                       herring_count.append(herring_counter)
                    
                    if stack_predections[0] ==1:
                        cv2.putText(img_raw, 'Detected Fish = Dab', (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 3)
                        if np.all (np.array(kliesche_count[-8:]) == 0) and np.all(np.array(herring_count[-8:]) == 0)  and np.all(np.array(kliesche_count[-8:]) == 0) and np.all(np.array(steinbutt_count[-8:]) == 0):
                            kliesche_counter+=1
                            kliesche_count.append(kliesche_counter)
                        kliesche_count.append(kliesche_counter)

                    if stack_predections[0] ==2:
                        cv2.putText(img_raw, 'Detected Fish = Turbot', (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 3)
                        if np.all (np.array(steinbutt_count[-8:]) == 0) and np.all(np.array(herring_count[-8:]) == 0)  and np.all(np.array(kliesche_count[-8:]) == 0) and np.all(np.array(steinbutt_count[-8:]) == 0):
                            steinbutt_counter+=1
                            steinbutt_count.append(steinbutt_counter)
                        steinbutt_count.append(steinbutt_counter)


                    if np.all(np.array(dummy_nums[-5:]) == 0):
                        counta+=1
                        dummy_nums.append(counta)
                    dummy_nums.append(counta)
                                    
                if (scores[i]*100) <80:
                    fine_class.append('no fish')                
                    dummy_nums.append(0)
                    cod_count.append(0)
                    kliesche_count.append(0)
                    herring_count.append(0)
                    steinbutt_count.append(0)
        print(dummy_nums)
        print(stack_predections)
        cv2.putText(img_raw, 'Total no of Fish = '+str(counta), (0,130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,  (0,0,255), 3)
        cv2.putText(img_raw, 'Total no of Cod = '+str(dorsch_counter), (0,160), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,  (0,0,255), 3)
        cv2.putText(img_raw, 'Total no of Dab = '+str(kliesche_counter), (0,180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,  (0,0,255), 3)
        cv2.putText(img_raw, 'Total no of Herring = '+str(herring_counter), (0,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,  (0,0,255), 3)
        cv2.putText(img_raw, 'Total no of Turbot = '+str(steinbutt_counter), (0,220), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,  (0,0,255), 3)
        if FLAGS.output:
            cv2.resize(img_raw, (650, 550))
            out.write(img_raw)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
