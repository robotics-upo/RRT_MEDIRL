// TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
// g++ -std=c++11 -shared rrt_star.cc -o rrt_star.so -fPIC -I $TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -D_GLIBCXX_USE_CXX_11_ABI=0 -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching && chmod +x *
// python rrt_net_test.py

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>


#define EPSILON 7
#define NUMNODES 10000
#define RADIUS  15
#define SQRT_2_2  (sqrt(2)/2)
#define EPS 0.0000000001

#include <stdlib.h>
#include <boost/concept_check.hpp>
#include <algorithm>

 
using namespace tensorflow;
using namespace cv;

REGISTER_OP("RRTStar")
    .Input("image: float")
    .Input("cost_map: float")
    .Input("label: float")
    .Input("random_f: float")
    .Output("path: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
    
    
class RRTStarOp : public OpKernel {
 public:
  explicit RRTStarOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);
    const Tensor& input_tensor3 = context->input(3);
    auto image_init = input_tensor1.flat<float>();
    auto cost_map_init = input_tensor2.flat<float>();
    auto random_init = input_tensor3.flat<float>();

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor1.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    int O = input_tensor1.shape().dim_size(0) ;
    int M = input_tensor1.shape().dim_size(1) ;
    int N = input_tensor1.shape().dim_size(2) ;
    
    
    Tensor _image(DT_FLOAT, TensorShape({O, N*M}));
    auto image = _image.matrix<float>();
    Tensor _cost_map(DT_FLOAT, TensorShape({O, N*M}));
    auto cost_map = _cost_map.matrix<float>();
    Tensor _out_map(DT_FLOAT, TensorShape({O, N*M}));
    auto out_map = _out_map.matrix<float>();
    Tensor _nodes_parent(DT_FLOAT, TensorShape({1,NUMNODES}));
    Tensor _nodes_x(DT_FLOAT, TensorShape({1,NUMNODES}));
    Tensor _nodes_y(DT_FLOAT, TensorShape({1,NUMNODES}));
    Tensor _nodes_cost(DT_FLOAT, TensorShape({1,NUMNODES}));
    auto nodes_parent =  _nodes_parent.matrix<float>();
    auto nodes_x =  _nodes_x.matrix<float>();
    nodes_x.setRandom();
    auto nodes_y =  _nodes_y.matrix<float>();
    nodes_y.setRandom();
    auto nodes_cost =  _nodes_cost.matrix<float>();
    int cont_goal = 0;
    int sum_goal_x = 0;
    int sum_goal_y = 0;
    int goal_x = 0;
    int goal_y = 0;    
    
    Tensor _random(DT_FLOAT, TensorShape({1,1}));
    auto random = _random.matrix<float>();

    // preparing the image and the cost_map

	for (int i = 0; i<random_init(0); i++ )
	{
	  random.setRandom();
	}


	
    for (int o = 0; o < O; o++) {      
      
      for(int i=0; i<N*M; i++)
      {
	    image(o,i) = (image_init(o*N*M+i)>0.55&&image_init(o*N*M+i)<0.65)||(image_init(o*N*M+i)>0.95&&image_init(o*N*M+i)<1.05)?0.0:image_init(o*N*M+i);
	    image(o,i) = image(o,i)>0.1?1.0:0.0; 
	  
	    cost_map(o,i) = cost_map_init(o*N*M+i);
	  
	    out_map(o,i) = 0.0;
      }
      
      Mat im_gray ;
	  im_gray.create(N, M, CV_32FC1);
	  Mat img_final;
      
            
      for(int i=0; i<N*M; i++)
      {
		 im_gray.at<float>(i) = image(o,i);
	  }
	  
	  dilate(im_gray, img_final, Mat(), Point(-1, -1), 2, 1, 1);

      for(int i=0; i<N*M; i++)
      {
		 image(o,i) = img_final.at<float>(i) ;
	  }	  
      
    }
    
    for (int o = 0; o < O; o++) {
      cont_goal = 0;
      sum_goal_x = 0;
      sum_goal_y = 0;
      nodes_x.setRandom();	
      nodes_y.setRandom();
      
      for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {     
		  if (image_init(o*N*M+i*M+j)>0.55&&image_init(o*N*M+i*M+j)<0.65)
		  {
			cont_goal++;
			sum_goal_x += i; sum_goal_y += j;
		  }	    
		    
		}
      }
      
      
      
      // Defining the goal
      if(cont_goal != 0)
      {  
		goal_x = sum_goal_x/cont_goal;
		goal_y = sum_goal_y/cont_goal;
      }
      else
      {
		goal_x = N/2;
		goal_y = M/2;
      }
      
      // first node and start
      nodes_parent(0) = 0;
      nodes_x(0) = N/2.0;
      nodes_y(0) = M/2.0;
      nodes_cost(0) = 0.0;
      
	       
      //~ // Loop begins
      for (int i = 1; i < NUMNODES; i++)
      {		

		nodes_x(i) = int(nodes_x(i)*(N-1.0));
		nodes_y(i) = int(nodes_y(i)*(M-1.0));
	 
		// if the point is filled, get another one
		while( image(o,nodes_x(i)*M+nodes_y(i))>0.5)
		{
		  random.setRandom();
		  nodes_x(i) = int(random(0)*(N-1.0));
		  random.setRandom();	
		  nodes_y(i) = int(random(0)*(M-1.0));
		}
		
	
		

		
		
		nodes_parent(i) = 0;
		nodes_cost(i) = 0;
		
		// find nearest point to the new node
		float dist_2_nn = (nodes_x(i)-nodes_x(0))*(nodes_x(i)-nodes_x(0))+(nodes_y(i)-nodes_y(0))*(nodes_y(i)-nodes_y(0));
		
		for (int j=0; j<i; j++)
		{
		  float dist_2_j = (nodes_x(i)-nodes_x(j))*(nodes_x(i)-nodes_x(j))+(nodes_y(i)-nodes_y(j))*(nodes_y(i)-nodes_y(j));
		  if (dist_2_nn > dist_2_j) 
		  {
			  dist_2_nn = dist_2_j;
			  nodes_parent(i) = j;
		  }
		}
		
		
		
		// step_from_to    
		if (dist_2_nn > EPSILON*EPSILON)
		{
		  float p_x = nodes_x(nodes_parent(i));
		  float p_y = nodes_y(nodes_parent(i));
		  nodes_x(i) = int(p_x + EPSILON*cos(atan2(nodes_y(i)-p_y,nodes_x(i)-p_x)));
		  nodes_y(i) = int(p_y + EPSILON*sin(atan2(nodes_y(i)-p_y,nodes_x(i)-p_x)));
		  
		  bool collisions = false;
		  float d = MAX(fabs(p_x-nodes_x(i)),fabs(p_y-nodes_y(i)))+1.0;
		  for( int k = 0; k<=d; k++)
		  {
			int img_x = k*(float((p_x-nodes_x(i)))/d)+nodes_x(i);
			int img_y = k*(float((p_y-nodes_y(i)))/d)+nodes_y(i);
			if(image(o,img_x*M+img_y)>0.5) 
			  collisions = true;
		  }
		  if(collisions)
		  {
			nodes_x(i) = int(nodes_x(nodes_parent(i)));
			nodes_y(i)= int(nodes_y(nodes_parent(i)));  
			dist_2_nn = 0;
		  }
		  else{
			dist_2_nn = EPSILON*EPSILON;
		  }
		}
		 
		
		
		// reWire
		float new_cost = (cost_map(o,nodes_x(i)*M+nodes_y(i))+cost_map(o,nodes_x(nodes_parent(i))*M+nodes_y(nodes_parent(i))))/2.0;
	    float node_cost_new = nodes_cost(nodes_parent(i)) + new_cost*sqrt(dist_2_nn);
		float new_node_parent = nodes_parent(i);
		
		for(int j = 1; j < i; j++)
		{
		  float dist_2_j = (nodes_x(i)-nodes_x(j))*(nodes_x(i)-nodes_x(j))+(nodes_y(i)-nodes_y(j))*(nodes_y(i)-nodes_y(j));
		  if( dist_2_j < RADIUS*RADIUS)
		  {
			float cost_j = (cost_map(o,nodes_x(i)*M+nodes_y(i))+cost_map(o,nodes_x(j)*M+nodes_y(j)))/2.0;
			float node_cost_j = nodes_cost(j) + cost_j*sqrt(dist_2_j);
			if(node_cost_j<node_cost_new)
			{
			  bool collisions = false;
			  float d = MAX(fabs(nodes_x(j)-nodes_x(i)),fabs(nodes_y(j)-nodes_y(i)))+1.0;
			  for( int k = 0; k<=d; k++)
			  {
				int img_x = (k*(float((nodes_x(j)-nodes_x(i)))/d)+nodes_x(i));
				int img_y = (k*(float((nodes_y(j)-nodes_y(i)))/d)+nodes_y(i));

				
				if(image(o,img_x*M+img_y)>0.5) 
				  collisions = true;
			  }
	 
			  if(!collisions)
			  {
				node_cost_new = node_cost_j;
				new_node_parent = j;
			  }    
			}	  
		  }
		}
		nodes_parent(i) = new_node_parent;
		nodes_cost(i) = node_cost_new;  
      }
      
      // draw function
      float dist_2_goal = (nodes_x(0)-goal_x)*(nodes_x(0)-goal_x)+(nodes_y(0)-goal_y)*(nodes_y(0)-goal_y);
      int nn = 0;
      
      for (int i=1; i<NUMNODES; i++)
      {
		float dist_2_goal_i = (nodes_x(i)-goal_x)*(nodes_x(i)-goal_x)+(nodes_y(i)-goal_y)*(nodes_y(i)-goal_y);
		if (dist_2_goal_i < dist_2_goal)
		{
		  dist_2_goal = dist_2_goal_i;
		  nn = i;
		}
      }
      

      while( nn != (nodes_parent(nn)) )
      {
		int d = MAX(fabs(nodes_x(nodes_parent(nn))-nodes_x(nn)),fabs(nodes_y(nodes_parent(nn))-nodes_y(nn)))+1.0;
		for (int i = 0; i < d; i++)
		{
		  int img_i = int(i*float(nodes_x(nodes_parent(nn))-nodes_x(nn))/(d)+nodes_x(nn));
		  int img_j = int(i*float(nodes_y(nodes_parent(nn))-nodes_y(nn))/(d)+nodes_y(nn));
		  out_map(o,img_i*M+img_j) = 1.0; 
		  image(o,img_i*M+img_j) =1.0; 		  
		}
		
		nn = nodes_parent(nn);
      }
	
      Mat out_gray ;
	  out_gray.create(N, M, CV_32FC1);
	  Mat _out_map_;
      
            
      for(int i=0; i<N*M; i++)
      {
		 out_gray.at<float>(i) = out_map(o,i);
	  }
	  
	  dilate(out_gray, _out_map_, Mat(), Point(-1, -1), 2, 1, 1);

      for(int i=0; i<N*M; i++)
      {
		 out_map(o,i) = _out_map_.at<float>(i) ;
	  }	 
    } 
         
    for (int i = 0; i < O*N*M; i++) {
       output_flat(i) = (out_map(i));
    } 
  }
};

REGISTER_KERNEL_BUILDER(Name("RRTStar").Device(DEVICE_CPU), RRTStarOp);
