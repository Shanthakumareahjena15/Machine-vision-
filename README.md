# Classification of fishe species using Model stacking approach  

To study the commercial fish stock, marine data is collected by preforming biological sampling. The conventional Biological sampling procedure includes sorting the catch into species, measuring the length and counting the number of the individual catch. This conventional technique in use is labour intensive and time consuming. To automate this process, we developed a hierarchical fish classification framework using convolution neural network, and machine learning classifiers to classify four different fish species.

# Datasets



We used two public and one custom dataset. The two public datasets are QUT FISH and Open image dataset. The examples in the public datasets are labelled with single label Fish and acts as the level one label of the semantic hierarchy. The custom dataset is captured in the laboratory at "Thünen-Institute (OF)" and in the fishery research vessel "Solea". Therefore, the dataset is named "Thünen dataset". The thünen dataset contains level one and level two label of the semantic hierarchy for each example in the dataset 

<figure>
<p float="left">
<img src="img/measuring_board.jpg" alt="drawing" width="300" height="100" />
<img src="img/her.jpg" alt="drawing" width="300" height="100" />
<img src="img/kliesche.jpg" alt="drawing" width="300" height="100" />
<img src="img/steinbutt.jpg" alt="drawing" width="300" height="200" />
  </p>
  <figcaption>(a) Cod (b) Herring (c) Dab (d) Turbot.</figcaption>
</figure>

<figure>
<img src="img/arch.JPG" alt="drawing">
<figcaption> Hierarchical annotation of the dataset</figcaption>
</figure>

# Architecture of the framework 
<figure>
<img src="img/class_pro.JPG" alt="drawing">
<figcaption>Architecture of the framework </figcaption>
</figure>

# Folder structure 
<figure>
<img src="img/folder_struct.JPG" alt="drawing">
<figcaption>Structuue of the files and folders </figcaption>
</figure>

# Gudeline to use the source code
1. Make sure the structure of the folders are correct as shown above 
2. Run the K-Fold.py script. It will generate set of csv files.
3. Run the stack_model.py. it will save the trained base and the meta model in the foler YOLO
4. Train the yolov3 netwotk and place the weigths inside YOLO/weights the folder 
5. place the testing video in the YOLO/data/videos folder
6. Run the detect_video.py file.
7. test video will be saved in the YOLO folder under the name output.avi

Note: detect_video.py scrtip takes multiple argument and the details is given in the readme file which is place inside YOLO folder 
