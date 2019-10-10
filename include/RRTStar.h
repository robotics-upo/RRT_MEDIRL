#ifndef RRT_STAR_H
#define RRT_STAR_H

#define RRT_STAR_INPUT_SHAPE_1 20
#define RRT_STAR_INPUT_SHAPE_2 200
#define RRT_STAR_INPUT_SHAPE_3 200
#define RRT_STAR_EPS 7
#define RRT_STAR_N_NODES 5000
#define RRT_STAR_RADIUS 12



#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <vector>
#include <experimental/optional>

#include <chrono>
#include <boost/concept_check.hpp>
#include <algorithm>
#include <random>

#include <iostream>
#include <stdio.h> 


#include <utils/random.hpp>
#include <utils/measures.hpp>
#include <utils/drawing.hpp>
#include <utils/timer.hpp>
#include <utils/fileactions.hpp>
#include <GeneralizedNode.h>
#include <GeneralizedVector3D.h>
#include <GeneralizedGraph.h>


using namespace tensorflow;

class RRTStarOp : public OpKernel {

		
		constexpr static int epsilon_ {RRT_STAR_EPS}; 
		constexpr static int n_nodes_ {RRT_STAR_N_NODES};
		constexpr static int radius_ {RRT_STAR_RADIUS};
		constexpr static int dim_image_i_ {RRT_STAR_INPUT_SHAPE_1};   // input_tensor1.shape().dim_size(0) ;
		constexpr static int dim_image_j_ {RRT_STAR_INPUT_SHAPE_2};  // input_tensor1.shape().dim_size(1) ;
		constexpr static int dim_image_k_ {RRT_STAR_INPUT_SHAPE_3};  // input_tensor1.shape().dim_size(2) ;
		
		constexpr static int dim_vector_  {dim_image_i_ * dim_image_j_ * dim_image_k_};
		
		using Vector3D = GeneralizedVector3D< dim_image_i_, dim_image_j_, dim_image_k_>;
		using Array3D = std::array< float, dim_image_i_* dim_image_j_* dim_image_k_>;
		using Vector2D = GeneralizedVector3D< 1, dim_image_j_, dim_image_k_>;
		using PointGoal = std::experimental::optional<cv::Point2i>;
		using VectGoals = std::vector<PointGoal>;	
		using Graph = GeneralizedGraph< RRT_STAR_INPUT_SHAPE_1, RRT_STAR_INPUT_SHAPE_2, RRT_STAR_INPUT_SHAPE_3, RRT_STAR_N_NODES, RRT_STAR_EPS, RRT_STAR_RADIUS>;
		
	public:
						
		Vector3D  map_;	
		Vector3D  goal_map_;
		Vector3D  cost_map_;
		Vector3D  out_map_;	
			
		Graph graph_;
		

	public:
		explicit RRTStarOp(OpKernelConstruction* context) : OpKernel(context) {};
	
		void Compute(OpKernelContext* context) override;
		
		void PreprocessInputsImage( const Tensor& image_Tensor, const Tensor& costmap_Tensor);
		void PreprocessMap( const Tensor& image_Tensor);
		void PreprocessGoalImg( const Tensor& image_Tensor);
		
		float ComputeDist( const cv::Point2i& p1, const cv::Point2i& p2);
		
		void FindGoalImg( Array3D& target, std::experimental::optional<cv::Point2i>& goal, int ini )	;	
		void RRTStarFunction();	
		
	private:
		Utils::Timer::Timer t1;
		
};



#endif
