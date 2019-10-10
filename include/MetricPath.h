#ifndef METRIC_PATH_H
#define METRIC_PATH_H


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cmath>

#include <ostream>

#include "RRTStar.h"




using namespace tensorflow;

class MetricPathOp : public OpKernel {
	private:	

		constexpr static int epsilon_ {RRT_STAR_EPS}; 
		constexpr static int n_nodes_ {RRT_STAR_N_NODES};
		constexpr static int radius_ {RRT_STAR_RADIUS};
		constexpr static int dim_image_i_ {RRT_STAR_INPUT_SHAPE_1};   // input_tensor1.shape().dim_size(0) ;
		constexpr static int dim_image_j_ {RRT_STAR_INPUT_SHAPE_2};  // input_tensor1.shape().dim_size(1) ;
		constexpr static int dim_image_k_ {RRT_STAR_INPUT_SHAPE_3};  // input_tensor1.shape().dim_size(2) ;

		using Graph = GeneralizedGraph< dim_image_i_, dim_image_j_, dim_image_k_, n_nodes_, epsilon_, radius_>;
		using Vector3D = GeneralizedVector3D< dim_image_i_, dim_image_j_, dim_image_k_>;

		Vector3D  map_;	
		Vector3D  goal_map_;		
		Vector3D  label_;	
		Vector3D  path_;	


		Graph label_path_;
		Graph img_path_;
		
		
		constexpr static double diss_threshold_ {0.02};
		constexpr static double m2_threshold_low_ {0.10};
		constexpr static double m2_threshold_med_ {0.25};
		constexpr static double m2_threshold_high_ {0.5};

		constexpr static float dist_min_to_goal_ {3.0};
		constexpr static int rep_mapping_path_ {dim_image_j_ * dim_image_k_ / 4};
			
		float grid_size_ {0.05}; // m / pixel	



	public:
		explicit MetricPathOp(OpKernelConstruction* context) : OpKernel(context) {};		
		void  Compute(OpKernelContext* context) override;
		
		void PreprocessInputs( const Tensor& _map_tensor, const Tensor& _img_tensor, const Tensor& _label_tensor );

		float ComputeDistanceBetweenPaths( Graph& _g1,  Graph& _g2, const int& _i);
		float ComputeDissimilarity( Graph& _img_path, Graph& _label_path, const int& _i);
		
};

#endif 
