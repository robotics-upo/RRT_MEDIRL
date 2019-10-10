#include "RRTStar.h"

using namespace tensorflow;


void RRTStarOp::Compute(OpKernelContext* context) 
{
	t1.tic();
		
	const Tensor& input_tensor1 = context->input(0);
	const Tensor& input_tensor2 = context->input(1);
	const Tensor& input_tensor3 = context->input(3);

	Utils::Random::AddRandomness( input_tensor3 );
	
	Tensor* output_tensor = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor1.shape(), &output_tensor));
	auto output_flat = output_tensor->flat<float>();

	PreprocessInputsImage( input_tensor1, input_tensor2);

	Utils::Drawing::Clean( out_map_ );		

	RRTStarFunction();

	out_map_.CopyToTensor(output_flat);
	std::cout<<"time   "<<t1.toc()<<std::endl;	   		

}


void RRTStarOp::PreprocessInputsImage( const Tensor& image_Tensor, const Tensor& costmap_Tensor)
{
	PreprocessMap(image_Tensor);
	PreprocessGoalImg( image_Tensor);
	
	cost_map_.CopyFromTensor( costmap_Tensor );	
	

}

void RRTStarOp::PreprocessMap( const Tensor& image_Tensor)
{
	map_.CopyFromTensor( image_Tensor );
	map_.FilterIf( [](float a){ return !(( a<0.05) or (a>0.55 and a<0.65));} );

	Utils::Drawing::DilateImage( map_);	 	
}

void RRTStarOp::PreprocessGoalImg( const Tensor& image_Tensor)
{
	goal_map_.CopyFromTensor( image_Tensor );
	goal_map_.FilterIf( [](float a){ return (a>0.55 && a<0.65);} );		 	
}




void RRTStarOp::RRTStarFunction()
{
		for( int i = 0; i < dim_image_i_ ; i++)
		{						
			std::string main_dir = std::string("./map_keys");
			std::string key1 = map_.GetKeyImg<  7, 10111>( i);	
			std::string key2 = map_.GetKeyImg< 25, 12653>( i);	
			std::string key3 = map_.GetKeyImg< 37, 13841>( i);	
			std::string rnd_n = std::to_string( Utils::Random::getRandomNumber<int>(1, 40)); 
			
			std::string dir = main_dir +"/";
			dir += rnd_n;				
			dir += "/";
			dir += key1;
			dir += "/";
			dir += key2;
			dir += "/";
			dir += key3;   		
		
			if( Utils::File::CreateDir( dir ))
			{
				graph_.InitGraph(i);
				graph_.FindParentByDistance( map_, i);
				graph_.SimplifyGraph( i );
				graph_.UpdateConnections( map_, i);
				graph_.Save( dir, i);
			}	
			else
			{
				graph_.Load( dir , i);
			}
			
			graph_.FindParentByCost( cost_map_, i);
		}
		
		Utils::Drawing::DrawPath( graph_, goal_map_, out_map_);			
		Utils::Drawing::DilateImage( out_map_);			
}







