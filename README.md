# RRT_MEDIRL
RRT_MEDIRL implementation for Tensorflow

## Getting Started

For more information about this work please check this paper: 

*Learning RRT\*-based human-aware path planning through Maximum Entropy Deep Inverse Reinforcement Learning* by G. Mier, N. Pérez-Higueras, F. Caballero and L. Merino. Submmited to the 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2018) 


The data used to train and test the model is [here (new data)](http://robotics.upo.es/datasets/irlrrt/) and [here (upo_fcn_learning package)](https://github.com/robotics-upo/upo_fcn_learning/tree/master/data). 

### Prerequisites

The code uses OpenCV, Numpy, Tensorflow and Keras.

To install the python dependencies:
```
sudo apt install python-pip
sudo pip install -r requirements.txt
```

[Opencv](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html) has to be installed to link with your gcc compiler.

### Installing

First, download the repo and create a bin folder on it:

```
cd <workspace>
git clone https://github.com/robotics-upo/RRT_MEDIRL.git
cd RRT_MEDIRL
mkdir bin
```

Change the variable "RRT_DIR" in the [compile.sh file](compile.sh) to have the dir of the bin folder. Then, compile the layers as:

```
sudo chmod +x compile.sh
./compile.sh
```

## Running the tests

Unfortunately, one of our biggest optimization is to compute the array size in compilation time.
To modify the parameters of the RRT*, change them in the first lines of the [RRTStar.h file](include/RRTStar.h).

To run the test, change the parameter **RRT_STAR_INPUT_SHAPE_1** to 1. Then, from the main folder, run:
```
python test_python_rrt.py
```
This will create two new images in the [resources](resources) folder. One with the path and other with the map and the path of the test example.

To train or evaluate a network (or predict using it), modify the values **train**, **evaluate** and **predict** in [rrt_train file](rrt_train.py). Values equal to 1 will execute that part of the code. Values equal to 0 won't.
Also, the value of **batch_size** in [rrt_train file](rrt_train.py) has to be equal than the **RRT_STAR_INPUT_SHAPE_1** in [RRTStar.h file](include/RRTStar.h).

As the loading of images is really slow, make sure the first time you run a new dataset in [the rrt_train file](rrt_train.py) as:

```
create_npy_files = True
load_npy_files = False
```

and the other times:

```
create_npy_files = False
load_npy_files = True
```

Lastly, in [the rrt_train file](rrt_train.py), variables **dataset_dir**, **testset_dir**, **save_dir1** and **save_dir2** should point to the dataset your going to use (right now pointing to the [data folder](data)).


## Structure and contents:
In the folders you will find data and scripts to reproduce our experiments and results. About folders and its contents:

### Matlab Code Folder:
The code to make the plots used in the paper are in this folder. All the code in this folder use Matlab 2017b.

### Results Folder

[Results folder](result) will be empty until you execute the [rrt_train file](rrt_train.py) in prediction mode.

### Resources Folder

[The Resources folder](resources) contains an example to test the path planning algorithm.

### Data Folder

[The data folder](data) should contain the dataset to use.


## Authors

* **Gonzalo Mier** - *Initial work* - [gonmiermu](https://github.com/gonmiermu)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
