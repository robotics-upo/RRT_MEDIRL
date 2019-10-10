#ifndef RRT_STAR_NODE_CLASS_H
#define RRT_STAR_NODE_CLASS_H

#include <experimental/optional>

#include "opencv2/opencv.hpp"

#include <GeneralizedVector3D.h>
#include <utils/measures.hpp>
#include <utils/random.hpp>

#include <sstream>
#include <string>


template<int X, int Y, int Z>
class GeneralizedNode{
	public: 
		cv::Point2i pos;
		std::experimental::optional<uint16_t> parent;
		float pos_cost {0.0};
		float acc_cost {0.0};
		
		struct aux_struct{
			size_t v_size;
			struct data
			{
				uint16_t  pos_x; 
				uint16_t  pos_y; 
				uint16_t parent;
				std::vector<uint16_t> connections;
			}data;
			};
		aux_struct data_comp;		
		
	public:	
		GeneralizedNode() = default;
		~GeneralizedNode() = default;
		
		void CreateRandomPoint( void )
		{
			pos = cv::Point2i( Utils::Random::getRandomNumber<int>(0,Y-1), Utils::Random::getRandomNumber<int>(0,Z-1));
		}

		void GetPosCost( GeneralizedVector3D<X,Y,Z>& _costmap, int _i )
		{
			if( parent )
				pos_cost = _costmap[ _i*Y*Z + pos.x*Z + pos.y];
		}
		
		auto GetNodeInfo()
		{
			data_comp.v_size = (data_comp.data.connections.size() + 3)*sizeof(uint16_t); 
			data_comp.data.pos_x = pos.x ;
			data_comp.data.pos_y = pos.y ;
			data_comp.data.parent = *parent ;
			return data_comp; 
		}		
		
		void SetNodeInfo()
		{			
		    pos.x = data_comp.data.pos_x;	
		    pos.y = data_comp.data.pos_y;	
		    parent = static_cast<int>(data_comp.data.parent);
		}



};	


#endif
