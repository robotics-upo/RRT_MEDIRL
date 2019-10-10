
#include "MetricPath.h"


#include <iostream>

using namespace tensorflow;

void MetricPathOp::Compute(OpKernelContext* context) 
{
	const Tensor& img_tensor = context->input(0);
    const Tensor& label_tensor = context->input(1);    
    const Tensor& map_tensor = context->input(2);
    const Tensor& v_const = context->input(3);
    

    Tensor* output_tensor1 = NULL;
    Tensor* output_tensor2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_const.shape(), &output_tensor1));
    OP_REQUIRES_OK(context, context->allocate_output(1, v_const.shape(), &output_tensor2));
    auto distance_paths = output_tensor1->flat<float>();
    distance_paths(0) = 0.0;
    auto dissimilarity = output_tensor2->flat<float>();
    dissimilarity(0) = 0.0;

	PreprocessInputs( map_tensor, img_tensor, label_tensor );

    int good_label_images {0};

	for (int i = 0; i < dim_image_i_; i++) 
	{	
		auto goal = goal_map_.FindGoalImg( i);
		std::cout<<*goal<<std::endl;
		if( goal )
		{
		
			cv::Point2i p_goal_img = path_.FindNearestNode( *goal, i);	
			cv::Point2i p_goal_lbl = label_.FindNearestNode( *goal, i);			
				
			if( Utils::Measures::ComputeDist( *goal, p_goal_lbl) <= dist_min_to_goal_)
			{
				good_label_images++;
				
				img_path_.ExtractPathFromVector3D( path_, p_goal_img, i);		
				label_path_.ExtractPathFromVector3D( label_, p_goal_lbl, i);		
					
				distance_paths(0) += ComputeDistanceBetweenPaths( img_path_, label_path_, i) * grid_size_;

				dissimilarity(0)  += ComputeDissimilarity( img_path_, label_path_, i);
			}	
		}
    }	
    
	distance_paths(0) /= good_label_images;
	dissimilarity(0)  /= good_label_images; 	
}



void MetricPathOp::PreprocessInputs( const Tensor& _map_tensor, const Tensor& _img_tensor, const Tensor& _label_tensor )
{
	map_.CopyFromTensor( _map_tensor );
	map_.FilterIf( [](float a){ return !(( a<0.05) or (a>0.55 and a<0.65));} );
	Utils::Drawing::DilateImage( map_);	 	
	
	goal_map_.CopyFromTensor( _map_tensor );
	goal_map_.FilterIf( [](float a){ return (a>0.55 && a<0.65);} );			

	path_.CopyFromTensor( _img_tensor );	
	label_.CopyFromTensor( _label_tensor );	 	
}	



float MetricPathOp::ComputeDistanceBetweenPaths( Graph& _g1,  Graph& _g2, const int& _i)
{
	return ( _g1.ComputeDistanceToAnotherGraph( _g2, _i)  + \
			 _g2.ComputeDistanceToAnotherGraph( _g1, _i)) / 2.0;
}


float MetricPathOp::ComputeDissimilarity( Graph& _img_path, Graph& _label_path, const int& _i)
{
	float result {0.0};
	int good_nodes {0};
	for( uint16_t j = 0; j < _label_path.nodes[_i].size()-1; j++)
	{	
		if(	 _label_path.nodes[_i][j].pos.x && _label_path.nodes[_i][j].pos.y )
		{
			result += (  _img_path.FindDistPointToPath( _label_path.nodes[_i][ j ].pos, _i) + 	    \
				     _img_path.FindDistPointToPath( _label_path.nodes[_i][j+1].pos, _i) ) * 	\
				     Utils::Measures::ComputeDist( _label_path.nodes[_i][j].pos, _label_path.nodes[_i][j+1].pos );
			good_nodes++;	     
		}
	}
	return static_cast<float>(result * std::pow( grid_size_ , 2.0) / (static_cast<float>( good_nodes)* 2.0));
}

