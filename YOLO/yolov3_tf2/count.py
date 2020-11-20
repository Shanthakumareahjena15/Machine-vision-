
import importlib.machinery


loader = importlib.machinery.SourceFileLoader('Object-Detection-API-master', 'D:\yolo_v3_object_detection\Object-Detection-API-master\module_test.py')
handle = loader.load_module('module_test.py')

#handle.module_test()