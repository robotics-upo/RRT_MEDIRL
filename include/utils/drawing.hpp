#ifndef DRAWING_H
#define DRAWING_H


#include "opencv2/opencv.hpp"

#include <GeneralizedNode.h>
#include <GeneralizedGraph.h>
#include <GeneralizedVector3D.h>
#include <utils/measures.hpp>


#include <iostream>


namespace Utils
{
namespace Drawing
{	
	using V_Goals = std::vector<std::experimental::optional<cv::Point2i>>;
	
	template<int X, int Y, int Z>
	void Clean( GeneralizedVector3D<X,Y,Z>& v )
	{ 
			v.FilterIf( [](float a){ return false;} );	
	}
	
	template<int X, int Y, int Z>
	void DrawLine( const GeneralizedNode<X,Y,Z>& n1, const GeneralizedNode<X,Y,Z>& n2, GeneralizedVector3D<X,Y,Z>& image, const int i)
	{
		if(n1.parent && n2.parent)
		{
			float d = Measures::ComputeDist(n1.pos, n2.pos);
			for (int pix = 0; pix < d; pix++)
			{
				cv::Point2i p = Measures::FindPointInLine(pix/d, n1.pos, n2.pos);
				image( i, p.x, p.y ) = 1.0; 
			}
		}
	}
	
	template< int X, int Y, int Z, int NODES, int EPS, int RADIUS>
	bool DrawPath( GeneralizedGraph< X, Y, Z, NODES, EPS, RADIUS>& _graph, GeneralizedVector3D<X,Y,Z>& _goalmap, GeneralizedVector3D<X,Y,Z>& _output, const int& i)
	{	
		const int max_path_rep_{ std::sqrt(Y*Z) };
		int path_rep{ 0 };
		std::experimental::optional<cv::Point2i> goal = _goalmap.FindGoalImg( i );
		int id;
		if( goal )
		{				
			id = _graph.FindNearestNode( *goal , i);
			
			while( id != *_graph.nodes[i][id].parent)
			{

				if( ++path_rep > max_path_rep_) { std::cout<<"Path circular"<<std::endl;      return false;}
				
				DrawLine( _graph.nodes[i][id], _graph.nodes[i][*_graph.nodes[i][id].parent], _output, i);
				id = *_graph.nodes[i][id].parent;				
			}
		}
		return true;
	}	
		
	
	template< int X, int Y, int Z, int NODES, int EPS, int RADIUS>
	void DrawPath( GeneralizedGraph< X, Y, Z, NODES, EPS, RADIUS>& _graph, GeneralizedVector3D<X,Y,Z>& _goalmap, GeneralizedVector3D<X,Y,Z>& _output)
	{		
		for(int i = 0; i < X; i++)
		{
			if( _graph.nodes[i].size()>2)
			DrawPath( _graph, _goalmap, _output, i);
		}	
	}
	

	
	template< int X, int Y, int Z, int NODES, int EPS, int RADIUS>
	void DrawFullTree( GeneralizedGraph< X, Y, Z, NODES, EPS, RADIUS>& _graph, GeneralizedVector3D<X,Y,Z>& _output, int _i)
	{				
		for( int id = 0; id < static_cast<int>(_graph.nodes[_i].size());id++)		
		{
			if( _graph.nodes[_i][id].parent)
			{
				DrawLine( _graph.nodes[_i][id], _graph.nodes[_i][*_graph.nodes[_i][id].parent], _output, _i);
			}
		}

	}
	
	template< int X, int Y, int Z, int NODES, int EPS, int RADIUS>
	void DrawFullTree( GeneralizedGraph< X, Y, Z, NODES, EPS, RADIUS>& _graph, GeneralizedVector3D<X,Y,Z>& _output)
	{		
		Clean( _output );
		for(int i = 0; i < X; i++)
		{
			 DrawFullTree( _graph, _output, i);
		}	
	}
		
	
	template< int X, int Y, int Z, int NODES, int EPS, int RADIUS>
	void DrawTreePoints( GeneralizedGraph< X, Y, Z, NODES, EPS, RADIUS>& _graph, GeneralizedVector3D<X,Y,Z>& _image)
	{
		Clean( _image );
		
		for(int i = 0; i < X; i++)
		{
			for( uint16_t id = 0; id < _graph.nodes[i].size();id++)		
			{
				_image( i, _graph.nodes[i][id].pos.x, _graph.nodes[i][id].pos.y ) = 1.0; 
			}
		}	
	}		
	
	
	template<int X, int Y, int Z>	
	void CopyVectToMat( GeneralizedVector3D<X,Y,Z>& a, cv::Mat& b, int id_img)
	{	
		for( int i = 0; i < Y*Z; i++)
			b.at<float>(i) = a.data[ std::fma( id_img, Y*Z, i)];	
	}

	template<int X, int Y, int Z>
	void CopyMatToVect( const cv::Mat& a, GeneralizedVector3D<X,Y,Z>& b, int id_img)
	{
		for( int i = 0; i < Y*Z; i++)
				b.data[ std::fma( id_img, Y*Z, i)] = a.at<float>(i);
	}		
	
	template<int X, int Y, int Z>
	void DilateImage( GeneralizedVector3D<X,Y,Z>&  image)
	{
		cv::Mat im_gray; 
		im_gray.create(Y, Z, CV_32FC1);

		for( int i=0; i<X; i++)
		{
			CopyVectToMat< X, Y, Z>(image, im_gray, i);
			dilate(im_gray, im_gray, cv::Mat(), cv::Point2i(-1, -1), 2, 1, 1);
			CopyMatToVect< X, Y, Z>(im_gray, image , i);	
		}	
	}
	
	template<int X, int Y, int Z>
	void DilateImage( GeneralizedVector3D<X,Y,Z>&  image, const int& i)
	{
		cv::Mat im_gray; 
		im_gray.create(Y, Z, CV_32FC1);

		CopyVectToMat< X, Y, Z>(image, im_gray, i);
		dilate(im_gray, im_gray, cv::Mat(), cv::Point2i(-1, -1), 2, 1, 1);
		CopyMatToVect< X, Y, Z>(im_gray, image , i);	

	}		

}
}

#endif
