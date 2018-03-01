// TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
// g++ -std=c++11 -shared metric_path.cc -o metric_path.so -fPIC -I $TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -D_GLIBCXX_USE_CXX_11_ABI=0 -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching && chmod +x *
// python rrt_net_test.py


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <ostream>

#define INF 99999999

#define diss_threshold_ 0.02;
#define m2_threshold_low_ 0.10;
#define m2_threshold_med_ 0.25;
#define m2_threshold_high_ 0.5;
#define rrt_repetitions_ 10;

using namespace tensorflow;
using namespace std;
using namespace cv;


REGISTER_OP("MetricPath")
    .Input("image: float")
    .Input("label: float")
    .Input("map: float")    
    .Input("v_const: float")         
    .Output("accuracy: float")  
    ;


class MetricPathOp : public OpKernel {
 public:
  explicit MetricPathOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);    
    const Tensor& input_tensor3 = context->input(2);
    const Tensor& v_const = context->input(3);
    auto image_init = input_tensor1.flat<float>();
    auto label_init = input_tensor2.flat<float>();
    auto map_init = input_tensor3.flat<float>();

    Tensor* output_tensor1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_const.shape(), &output_tensor1));
    auto accuracy = output_tensor1->flat<float>();

    
    int O = input_tensor1.shape().dim_size(0);
    int M = input_tensor1.shape().dim_size(1);
    int N = input_tensor1.shape().dim_size(2);
    
    Tensor _image(DT_FLOAT, TensorShape({O, N*M}));
    auto image = _image.matrix<float>();
    Tensor _img(DT_FLOAT, TensorShape({O, N*M}));
    auto img = _img.matrix<float>();
    Tensor _label(DT_FLOAT, TensorShape({O, N*M}));
    auto label = _label.matrix<float>(); 
    Tensor _lbl(DT_FLOAT, TensorShape({O, N*M}));
    auto lbl = _lbl.matrix<float>();    
    
        
    float acc = 0.0;
    float dis = 0.0;
    
    int cont_goal = 0;
    int sum_goal_x = 0;
    int sum_goal_y = 0;
    int goal_x = 0;
    int goal_y = 0;   
    int start_x = M/2;
    int start_y = N/2;
    
    
	Mat im_gray ;
	im_gray.create(N, M, CV_32FC1);
	Mat img_final;

    for( int o = 0; o < O; o++)
    {		
		for(int i=0; i<N*M; i++)
		{
		  img(o,i) = image_init(o*N*M+i);
		  lbl(o,i) = label_init(o*N*M+i);
		}		
	}
	int good_label = 0;    
	int all_label = 0;    
    for (int o = 0; o < O; o++) {
		
		bool flag_bad_label = false;
		vector<int> v_nodes_img_x;
		vector<int> v_nodes_img_y;
		vector<int> v_nodes_label_x;
		vector<int> v_nodes_label_y;
		vector<int> nodes_img_x;
		vector<int> nodes_img_y;
		vector<int> nodes_label_x;
		vector<int> nodes_label_y;	

		cont_goal = 0;
		sum_goal_x = 0;
		sum_goal_y = 0;
		

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {     
			  if (map_init(o*N*M+i*M+j)>0.55&&map_init(o*N*M+i*M+j)<0.65)
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
			flag_bad_label = true;
		}
				
		int it, i, j;		
		float dist_goal_img = INF;
		float dist_goal_lbl = INF;
		int near_p_img_x;
		int near_p_img_y;
		int near_p_lbl_x;
		int near_p_lbl_y;

		for( i = 0; i < N; i++)
		{
			for( j = 1; j < M; j++)
			{
				
				if (img(o*N*M+i*M+j)>0.5)
				{
					v_nodes_img_x.push_back(i);
					v_nodes_img_y.push_back(j);
					float dist_img_i = sqrt((float)((goal_x-i)*(goal_x-i)+(goal_y-j)*(goal_y-j)));
					if(dist_img_i< dist_goal_img) 
					{
						dist_goal_img = dist_img_i;
						near_p_img_x = i;
						near_p_img_y = j;
					}
				}		
				
				if (lbl(o*N*M+i*M+j)>0.5)
				{
					v_nodes_label_x.push_back(i);
					v_nodes_label_y.push_back(j);	
					float dist_lbl_i = sqrt((float)((goal_x-i)*(goal_x-i)+(goal_y-j)*(goal_y-j)));
					if(dist_lbl_i< dist_goal_lbl) 
					{
						dist_goal_lbl = dist_lbl_i;
						near_p_lbl_x = i;
						near_p_lbl_y = j;
					}						
				}								
			}
		}
		
		if(dist_goal_lbl>=3.0) flag_bad_label = true;
		
		all_label++;
		if(flag_bad_label==false)
		{
			good_label++;
			
			v_nodes_img_x.push_back(near_p_img_x);
			v_nodes_img_y.push_back(near_p_img_y);
			v_nodes_label_x.push_back(near_p_lbl_x);
			v_nodes_label_y.push_back(near_p_lbl_y);	
		
			
			nodes_img_x.push_back(v_nodes_img_x[0]);
			nodes_img_y.push_back(v_nodes_img_y[0]);
			
			v_nodes_img_x.erase(v_nodes_img_x.begin()+0);
			v_nodes_img_y.erase(v_nodes_img_y.begin()+0);
			int i_min_d = 0;
			i = 0;
			float dist_start=INF;
			
			while(dist_start>=1.0)
			{
				int p_x = nodes_img_x[i];
				int p_y = nodes_img_y[i];
				float dist = INF;
				dist_start=INF;

				for( j = 0; j < v_nodes_img_x.size(); j++)
				{
					float dist_j = sqrt((float)((p_x-v_nodes_img_x[j])*(p_x-v_nodes_img_x[j])+(p_y-v_nodes_img_y[j])*(p_y-v_nodes_img_y[j])));
					if( dist_j<dist)
					{
						dist = dist_j;
						i_min_d = j;
						dist_start = sqrt((float)((start_x-v_nodes_img_x[j])*(start_x-v_nodes_img_x[j])+(start_y-v_nodes_img_y[j])*(start_y-v_nodes_img_y[j])));
					}
					else if(dist_j==dist)
					{
						float dist_start_j = sqrt((float)((start_x-v_nodes_img_x[j])*(start_x-v_nodes_img_x[j])+(start_y-v_nodes_img_y[j])*(start_y-v_nodes_img_y[j])));
						if (dist_start_j < dist_start)
						{
							dist = dist_j;
							i_min_d = j;
							dist_start = dist_start_j;						
						}				
					}
				}
				
				nodes_img_x.push_back(v_nodes_img_x[i_min_d]);
				nodes_img_y.push_back(v_nodes_img_y[i_min_d]);
				v_nodes_img_x.erase(v_nodes_img_x.begin()+i_min_d);
				v_nodes_img_y.erase(v_nodes_img_y.begin()+i_min_d);		
				i++;
			}
			nodes_label_x.push_back(v_nodes_label_x[0]);
			nodes_label_y.push_back(v_nodes_label_y[0]);
			
			v_nodes_label_x.erase(v_nodes_label_x.begin()+0);
			v_nodes_label_y.erase(v_nodes_label_y.begin()+0);
			i_min_d = 0;
			i = 0;
			dist_start = INF;
			while(dist_start>=1.0)
			{
				int p_x = nodes_label_x[i];
				int p_y = nodes_label_y[i++];
				float dist = INF;			
				dist_start = INF;
				
				for( j = 0; j < v_nodes_label_x.size(); j++)
				{
					float dist_j = sqrt((float)((p_x-v_nodes_label_x[j])*(p_x-v_nodes_label_x[j])+(p_y-v_nodes_label_y[j])*(p_y-v_nodes_label_y[j])));
					if( dist_j<dist)
					{
						dist = dist_j;
						i_min_d = j;
						dist_start = sqrt((float)((start_x-v_nodes_label_x[j])*(start_x-v_nodes_label_x[j])+(start_y-v_nodes_label_y[j])*(start_y-v_nodes_label_y[j])));
					}
					else if(dist_j==dist)
					{
						float dist_start_j = sqrt((float)((start_x-v_nodes_label_x[j])*(start_x-v_nodes_label_x[j])+(start_y-v_nodes_label_y[j])*(start_y-v_nodes_label_y[j])));
						if (dist_start_j < dist_start)
						{
							dist = dist_j;
							i_min_d = j;
							dist_start = dist_start_j;						
						}				
					}
				}
				
				nodes_label_x.push_back(v_nodes_label_x[i_min_d]);
				nodes_label_y.push_back(v_nodes_label_y[i_min_d]);
				v_nodes_label_x.erase(v_nodes_label_x.begin()+i_min_d);
				v_nodes_label_y.erase(v_nodes_label_y.begin()+i_min_d);		
			}	

			
			
			
			int nodes_img_len = nodes_img_x.size();
			int nodes_label_len = nodes_label_x.size();			
			
					
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

							
			float dist_m_img = 0.0;
			for( i=0; i< nodes_img_len-1; i++)
			{
				int nearest_point_x = INF;
				int nearest_point_y = INF;
				float nearest_point_dist = INF;
				int p_x = nodes_img_x[i];
				int p_y = nodes_img_y[i];						
				for( j=0; j< nodes_label_len-1; j++)
				{
					float dist = sqrt((p_x-nodes_label_x[j])*(p_x-nodes_label_x[j])+(p_y-nodes_label_y[j])*(p_y-nodes_label_y[j]));
					if (dist < nearest_point_dist) 
					{
						nearest_point_dist = dist;
					}

				}
				dist_m_img += (nearest_point_dist);
			}
			dist_m_img /= (nodes_img_len-1);	

			float dist_m_label = 0.0;
			for( i=0; i< nodes_label_len-1; i++)
			{
				int nearest_point_x = INF;
				int nearest_point_y = INF;
				float nearest_point_dist = INF;
				int p_x = nodes_label_x[i];
				int p_y = nodes_label_y[i];						
				for( j=0; j< nodes_img_len-1; j++)
				{
					float dist = sqrt((p_x-nodes_img_x[j])*(p_x-nodes_img_x[j])+(p_y-nodes_img_y[j])*(p_y-nodes_img_y[j]));
					if (dist < nearest_point_dist) 
						nearest_point_dist = dist;
				}

				dist_m_label += (nearest_point_dist);
			}
			dist_m_label /= (nodes_label_len-1);			
			
			acc += dist_m_img + dist_m_label;
		}
	}
		
  accuracy(0) = acc/2.0/good_label*0.05;
    
  }
};

REGISTER_KERNEL_BUILDER(Name("MetricPath").Device(DEVICE_CPU), MetricPathOp);
