#!/bin/bash

#source activate tensorflow   

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
RRT_DIR=$(echo './bin')

rm -R bin
mkdir -p bin
cd bin

g++ -std=c++17 -shared ../src/RRTStar.cc -o RRTStar.so -I ../include 			 \
		-O3 -march=native -Wall -Werror -Wfatal-errors -Wl,--no-undefined	    	 \
		-fPIC -I$TF_INC -I $TF_INC/external/nsync/public 		         \
		-L$TF_LIB -I /usr/local/include/opencv -I /usr/local/include/opencv2 	 \
		-L /usr/local/lib/ -ltensorflow_framework 			         \
		-D_GLIBCXX_USE_CXX11_ABI=0 -lopencv_core -lopencv_imgproc 	         \
		-lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d          \
		-lopencv_calib3d -lopencv_objdetect -lopencv_stitching -lstdc++fs  

		
g++ -std=c++17 -shared ../src/MetricPath.cc -o MetricPath.so -I ../include  	         \
		-O3 -march=native -Wall -Werror -Wfatal-errors -Wl,--no-undefined	    	 \
		-fPIC -I$TF_INC -I $TF_INC/external/nsync/public 			 \
		-L$TF_LIB -I /usr/local/include/opencv -I /usr/local/include/opencv2 	 \
		-L /usr/local/lib/ -ltensorflow_framework 			         \
		-D_GLIBCXX_USE_CXX11_ABI=0 -lopencv_core -lopencv_imgproc 	         \
		-lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d          \
		-lopencv_calib3d -lopencv_objdetect -lopencv_stitching	-lstdc++fs 

chmod +x *
			
		
g++ -std=c++17 -shared ../src/rrt_star.cc -o rrt_star.so $RRT_DIR/RRTStar.so -I ../include  	 \
		-O3  -march=native -Wall -Werror -Wfatal-errors -Wl,--no-undefined	    	 \
		-fPIC -I$TF_INC -I $TF_INC/external/nsync/public 		         \
		-L$TF_LIB -I /usr/local/include/opencv -I /usr/local/include/opencv2 	 \
		-L /usr/local/lib/ -ltensorflow_framework 			         \
		-D_GLIBCXX_USE_CXX11_ABI=0 -lopencv_core -lopencv_imgproc 		 \
		-lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d 	 \
		-lopencv_calib3d -lopencv_objdetect -lopencv_stitching	-lstdc++fs 
		
		
g++ -std=c++17 -shared ../src/metric_path.cc -o metric_path.so $RRT_DIR/MetricPath.so -I ../include   \
		-O3 -march=native -Wall -Werror -Wfatal-errors -Wl,--no-undefined	    	    \
		-fPIC -I$TF_INC -I $TF_INC/external/nsync/public 		            \
		-L$TF_LIB -I /usr/local/include/opencv -I /usr/local/include/opencv2 	    \
		-L /usr/local/lib/ -ltensorflow_framework 			            \
		-D_GLIBCXX_USE_CXX11_ABI=0 -lopencv_core -lopencv_imgproc 	            \
		-lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d             \
		-lopencv_calib3d -lopencv_objdetect -lopencv_stitching	-lstdc++fs  
		

chmod +x *

cd ..
