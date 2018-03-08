# RRT_MEDIRL
RRT_MEDIRL implementation for Tensorflow

## Information
For more information about this work please check paper: 

*Learning RRT\*-based human-aware path planning through Maximum Entropy Deep Inverse Reinforcement Learning* by G. Mier, N. Pérez-Higueras, F. Caballero and L. Merino. Submmited to the 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2018) 

The data used to train and test the model is in [here (new data)](http://robotics.upo.es/datasets/irlrrt/) and [here (upo_fcn_learning package)](https://github.com/robotics-upo/upo_fcn_learning/tree/master/data). The code uses OpenCV, Numpy, Tensorflow and Keras.



## To test the code:

```
cd <workspace>
git clone https://github.com/robotics-upo/RRT_MEDIRL.git
cd RRT_MEDIRL
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared rrt_star.cc -o rrt_star.so -fPIC -I $TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -D_GLIBCXX_USE_CXX_11_ABI=0 -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching && chmod +x *
 g++ -std=c++11 -shared metric_path.cc -o metric_path.so -fPIC -I $TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -D_GLIBCXX_USE_CXX_11_ABI=0 -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching && chmod +x *
rrt_train_evaluate_and_predict.py
```



## Structure and contents:
In the folders you will find data and scripts to reproduce our experiments and results. About folders and its contents:

### code Folder:
The code to make the plots used in the paper are in this folder. All the code in this folder use Matlab 2017b.

### result, labels and rrt_out Folders:
In those folders, the rrt_predict.py and rrt_train_evaluate_and_predict.py save the outputs.The csv and image labels are saved in the *labels* folder. The csv and image of the output of the RRT_MEDIRL are saved in the *rrt_out* folder. Images of the map, the output of the RRT_MEDIRL and the labels are saved in the *result* folder.


### rrt_star.cc:
A c++ file that computes the rrt\* using a costmap and a map. It is necessary to compile this code before to execute any of the python code.

### metric_path.cc:
A c++ file that computes the metrics to compare the label with the output of the RRT_MEDIRL. It is necessary to compile this code before to execute any of the python code.


### rrt_train.py:
To launch the script, you can use:
```
python rrt_train.py
```
This program will train the RRT_MEDIRL network and save on *estimator*.


### rrt_predict.py:
To launch the script, you can use:
```
python rrt_predict.py
```
This program will use the trained model of the RRT_MEDIRL network (*estimator*) to predict the output with the test data.

### rrt_evaluate.py:
To launch the script, you can use:
```
python rrt_evaluate.py
```
When rrt_evaluate.py is executed, it uses the trained model of the RRT_MEDIRL network (*estimator*) to compute and save some metrics of the output of the net.

### rrt_train_evaluate_and_predict.py
To launch the script, you can use:
```
python rrt_train_evaluate_and_predict.py
```
This program does the same as rrt_train.py, rrt_predict.py and rrt_evaluate.py, but just in one file




