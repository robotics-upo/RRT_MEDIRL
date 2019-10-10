#ifndef RRT_STAR_GENERIC_VECTOR3D_H
#define RRT_STAR_GENERIC_VECTOR3D_H

#include <tensorflow/core/framework/op.h>
#include "opencv2/opencv.hpp"

#include <vector>
#include <forward_list>
#include <bitset>

#include <algorithm>
#include <random>

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <utils/measures.hpp>
#include <utils/random.hpp>



template<int X, int Y, int Z>
class GeneralizedVector3D
{
	
	public:	
		std::array<float, X*Y*Z> data;		
		
		const std::forward_list<cv::Point2i> boundary_ { cv::Point2i( +1,  0), 	\
												 cv::Point2i( +1, -1), 	\
												 cv::Point2i(  0, -1), 	\
												 cv::Point2i( -1, -1), 	\
												 cv::Point2i( -1,  0), 	\
												 cv::Point2i( -1, +1),	\
												 cv::Point2i(  0, +1), 	\
												 cv::Point2i( +1, +1)};		
		
	public:
		float& operator()( int i, int j, int k)
		{
			return data[ i * Y * Z + j * Z + k];
		}
		float& operator()( int i, int j)
		{
			return data[ i * Y * Z + j];
		}	
		float& operator()( int i)
		{
			return data[ i];
		}		
		float& operator[]( int i)
		{
			return data[ i];
		}									
		void CopyFromTensor(const tensorflow::Tensor& t)
		{
			auto t_float = t.flat<float>();
			for( int i=0; i<X*Y*Z; i++)
				data[i] = t_float(i); 
		}
		void CopyToTensor(auto& t)
		{
			for( int i=0; i<X*Y*Z; i++)
				t(i) = std::max(float(0.0),std::min(float(1.0),data[i])); 
		}			
		void FilterIf(std::function<bool (float& )> f)
		{
			for(auto& p: data)
				p = f(p)?1.0:0.0;
		}
		void ApplyFunc(std::function<void (float& )> f)
		{
			std::for_each(data.begin(), data.end(), f );
		}
		void ApplyFuncEachImg(std::function<void ( std::array<float, X*Y*Z>&, int )> f)
		{	
			for( int i=0; i<X; i++)
				f( data, i);
		}
		bool isColliding( cv::Point2i& p, int img )
		{
			return data[img* Y*Z + p.x *Z + p.y] > 0.5;
		}
		
		bool isColliding( const cv::Point2i& p1, const cv::Point2i& p2, int img ) const
		{
			bool collisions = false;				
						
			float d = Utils::Measures::ComputeDist(p1, p2);
			if( d > 0)
			{
				for( int k = 0; k<=d; k++)
				{
					cv::Point2i p = Utils::Measures::FindPointInLine(k/d, p1, p2);
					if( data[img* Y*Z + p.x *Z + p.y] > 0.5 ) 
						collisions = true;
				}
			}	
			return collisions;
		}		
		
		std::experimental::optional<cv::Point2i> FindGoalImg( int img )
		{
			std::experimental::optional<cv::Point2i> goal;	
			int cont_goal {0};
			cv::Point2i sum_goal { 0 , 0};

			for( int j = 0; j < Y; j++)
				for( int k = 0; k < Z; k++)
					if ( data[ img * Y*Z  + j * Z + k ] > 0.5)
					{
						cont_goal++;
						sum_goal.x += j; sum_goal.y += k;
					}
			if(cont_goal != 0)
			{  
				goal = cv::Point2i(sum_goal.x/cont_goal, sum_goal.y/cont_goal);
			}
			return goal;
		}
		
		cv::Point2i FindNearestNode( const cv::Point2i& _goal, int& _i)
		{
			float min_dist = FLT_MAX;
			cv::Point2i p_nearest;
			for( int j = 0; j < Y ; j++)
			{
				for( int k = 0; k < Z ; k++)
				{
					auto p = cv::Point2i(j,k);
					if( isColliding( p, _i ))
					{
						float dist = Utils::Measures::ComputeDist({j,k}, _goal);
						if( dist < min_dist)
						{
							min_dist = dist;
							p_nearest = {j,k};
						}
					}	
				}			
			}
			return p_nearest;
		}
								
		void LengthsPathToStart( std::vector<int>& _lengths, const int& _i)
		{    
			_lengths[  (Y / 2) * Z + (Z / 2) ] = 1;	
			for( int rep = 0; rep < Y*Z/9  ; rep++)
			{
				for( int j = 1; j < Y-1 ; j++)
				{
					for( int k = 1; k < Z-1 ; k++)
					{
						for( auto& b: boundary_ )
							UpdateLengthsImg( cv::Point2i( j , k ), b , _lengths,  _i);					
					}
				}
			}
		}

		void UpdateLengthsImg( const cv::Point2i& _p_c, const cv::Point2i& _move, std::vector<int>& _lengths, const int& _i)
		{
			if( data[ _i *Y*Z + _p_c.x *Z + _p_c.y] > 0.1 && \
				_lengths[ _p_c.x *Z + _p_c.y] > 0 )
			{
				auto p_n = _p_c + _move;
				if( data[ _i *Y*Z + p_n.x *Z + p_n.y] > 0.1 && \
					_lengths[ p_n.x *Z + p_n.y ] < 1 )
				{
					_lengths[ p_n.x *Z + p_n.y ] = _lengths[ _p_c.x *Z + _p_c.y ] + 1;					
				}	
			}
		}
		
		
		
		template< int N, int PRIME>
		std::string GetKeyImg(int _i )
		{
			std::string key1;
			std::string key2;
			int key3 = 0;
			std::stringstream res1;
			std::stringstream res2;
			for( int j = 0; j < Y*Z; j = j+N)
			{
				int byte = 0;
				for( int k = 0; k < N; k++)
				{
					if( j + k < Y * Z)
					{
						byte += static_cast<int>(data[ _i * Y*Z + j + k ])<<k;
					}
				} 
				res1 << std::hex << std::uppercase << std::bitset<N>(byte).to_ulong();
			}
			key1 = res1.str();
			
			std::string hex_chars {"0123456789ABCDEF"};  
			for(char c: hex_chars)
			{
				res2 << std::count(key1.begin(), key1.end(), c);
				key2 += res2.str();
			}
			for(char c: key2)
			{
				key3 += c*PRIME; //Big prime number
			}					
			
			return  std::to_string(key3);
		}
							
					
};

	
#endif
