## Important Points to rember 
The pixel location of the measuring board has to  be computed in prior and mention  it in the detect.py and detect_video.py  



## Command Line Args Reference

```bash
load_weights.py:
  --output: path to output
    (default: './weights/yolov3.tf')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './weights/yolov3.weights')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)

detect.py:
  --classes: path to classes file
    (default: './data/labels/coco.names')
  --images: path to input images as a string with images separated by ","
    (default: 'data/images/dog.jpg')
  --output: path to output folder
    (default: './detections/')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './weights/yolov3.tf')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)

detect_video.py:
  --classes: path to classes file
    (default: './data/labels/coco.names')
  --video: path to input video (use 0 for webcam)
    (default: './data/video/paris.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './weights/yolov3.tf')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)
```

## Acknowledgments
* [Yolov3 TensorFlow 2 Amazing Implementation](https://github.com/zzh8829/yolov3-tf2)
* [Another Yolov3 TensorFlow 2](https://github.com/heartkilla/yolo-v3)
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
* [A Tensorflow Slim implementation](https://github.com/mystic123/tensorflow-yolo-v3)
* [The AI Guy](https://github.com/theAIGuysCode)
