#ifndef RRT_STAR_GENERIC_GRAPH_H
#define RRT_STAR_GENERIC_GRAPH_H

#include <vector>
#include <cfloat>


#include <GeneralizedNode.h>
#include <GeneralizedVector3D.h>
#include <utils/random.hpp>
#include <utils/timer.hpp>
#include <utils/fileactions.hpp>



template< int X, int Y, int Z, int NODES, int EPS, int RADIUS>
class GeneralizedGraph{
	using Node = GeneralizedNode<X,Y,Z>;

	public: 
		std::vector<std::vector<Node>> nodes = std::vector<std::vector<Node>>(X, std::vector<Node>(0,Node()));		
		
		const std::forward_list<cv::Point2i> boundary_ { cv::Point2i( +1,  0), 	\
												 cv::Point2i( +1, -1), 	\
												 cv::Point2i(  0, -1), 	\
												 cv::Point2i( -1, -1), 	\
												 cv::Point2i( -1,  0), 	\
												 cv::Point2i( -1, +1),	\
												 cv::Point2i(  0, +1), 	\
												 cv::Point2i( +1, +1)};		
	public:
		

		void InitGraph(int _i)
		{
			nodes[_i].clear();
			for( int j = 0; j < NODES; j++ )
			{				
				nodes[_i].emplace_back();				
				nodes[_i][j].CreateRandomPoint();
			}
			Shuffle(_i);
		
			nodes[_i][0].pos = {Y/2,Z/2};
			nodes[_i][0].parent = 0;
		}
		
		void InitGraph()
		{
			for( int i = 0; i < X; i++)
				InitGraph(i);
		}
						
				
		void Shuffle( int _i) 
		{
				std::shuffle(nodes[_i].begin(), nodes[_i].end(), Utils::Random::getGenerator());
		}		
		
		int FindNearestNode( const cv::Point2i& _goal, int _i)
		{
			float min_dist = FLT_MAX;
			int id_nearest = 0;
			for( int j = 0; j < NODES; j++ )
			{
				if( nodes[_i][j].parent) 
				{
					float dist = Utils::Measures::ComputeDist(nodes[_i][j].pos, _goal);
					if( dist < min_dist)
					{
							min_dist = dist;
							id_nearest = j;
					}	
				}			
			}
			return id_nearest;
		}
		
		float FindDistPointToPath(const cv::Point2i& _p1, int _i)
		{
			float dist  = std::numeric_limits<float>::max();
	
			for( const auto& p2: nodes[_i])
			{
				float dist_p1_p2 = Utils::Measures::ComputeDist( _p1, p2.pos);
				if( dist_p1_p2 < dist)
					dist = dist_p1_p2;
			}
			return dist;	
		}
		
		float ComputeDistanceToAnotherGraph(const GeneralizedGraph< X, Y, Z, NODES, EPS, RADIUS>& _g, const int& _i)
		{
			float dist {0.0};
			
			for( const auto& p1: _g.nodes[_i])
			{
				dist += FindDistPointToPath( p1.pos, _i);
			}
			return dist/_g.nodes[_i].size();
		}
		

		
		void FindParentByDistance( const GeneralizedVector3D<X,Y,Z>& _image, int _i)
		{
			for( int j = 1; j < static_cast<int>(nodes[_i].size()); j++)
			{
				FindParentByDistance( _image, nodes[_i][j], j, _i);
			}
		}	
		
		void FindParentByDistance( const GeneralizedVector3D<X,Y,Z>& _image)
		{
			for( int i = 0; i < X; i++)
			{
				FindParentByDistance( _image, i);
			}
		}				
		
		void FindParentByDistance( const GeneralizedVector3D<X,Y,Z>& _image, Node& _n, const int& _id_n, const int& _i)
		{
			float min_dist = FLT_MAX;
			cv::Point2i new_pos = _n.pos;
			cv::Point2i next_pos = _n.pos;
			for( int j = 0; j < _id_n; j++ )
			{
				if( nodes[_i][j].parent) 
				{
					float dist = Utils::Measures::ComputeDist(nodes[_i][j].pos, _n.pos);
					if( dist < min_dist)
					{
						if( dist > EPS)
						{
							new_pos = Utils::Measures::GetPointNear<int, EPS>(nodes[_i][j].pos,  _n.pos);
						}
						else
						{
							new_pos = _n.pos;
						}
						if( !_image.isColliding(nodes[_i][j].pos,new_pos,_i))
						{
							min_dist = dist;
							_n.parent = j;
							next_pos = new_pos;
						}
					}	
				}
				if( min_dist < 0.1 ) // Same position as another node
				{
					break;
				}				
			}
			_n.pos = next_pos;
			if( min_dist < 0.1 ) 
			{
				_n.parent = std::experimental::nullopt;
			}
		}
		
		void FindParentByCost( GeneralizedVector3D<X,Y,Z>& _costmap)
		{
			for( int i = 0; i < X; i++)
			{
				FindParentByCost( _costmap, i);
			}
		}
		
		void FindParentByCost( GeneralizedVector3D<X,Y,Z>& _costmap, const int& _i)
		{
			GetCostsFromMap( _costmap, _i);	
			UpdateParentCosts( _i );		
			UpdateParentCosts( _i );		

			for( size_t j = 1; j < nodes[_i].size(); j++)
			{
				FindParentByCost( nodes[_i][j], j, _i);
			}
		}				
		
		void FindParentByCost( Node& _n, const int& _id_n, const int& _i)
		{
			float min_cost = nodes[_i][_id_n].acc_cost;
			float cost;

			for( auto& j: nodes[_i][_id_n].data_comp.data.connections )
			{
				float dist = Utils::Measures::ComputeDist( nodes[_i][j].pos, _n.pos);
				cost = ComputeCost( nodes[_i][j], _n, dist);
				
				if( cost < min_cost)
				{		
					min_cost = cost;
					_n.parent = j;

				}
			}
			_n.acc_cost = min_cost;
		}
		
		void UpdateParentCosts( int _i )
		{
			for( size_t j = 1; j < nodes[_i].size(); j++)
			{
				if( nodes[_i][j].parent )
				{
					nodes[_i][j].acc_cost = ComputeCost( nodes[_i][*nodes[_i][j].parent], nodes[_i][j], Utils::Measures::ComputeDist( nodes[_i][j].pos, nodes[_i][*nodes[_i][j].parent].pos));
				}
				else
				{
					nodes[_i][j].acc_cost = FLT_MAX;
				}
			}
		}
				
		
		void GetCostsFromMap( GeneralizedVector3D<X,Y,Z>& _costmap)
		{
			for( int i = 0; i < X; i++)
			{
				GetCostsFromMap( _costmap, i);
			}
		}
		
		void GetCostsFromMap( GeneralizedVector3D<X,Y,Z>& _costmap, const int& _i)
		{
			for( size_t j = 1; j < nodes[_i].size(); j++)
			{		
				nodes[_i][j].GetPosCost( _costmap, _i );
			}		
		}		
	
		
		void ExtractPathFromVector3D( GeneralizedVector3D<X, Y, Z>& _image, cv::Point2i& _goal, int& _i )
		{	
			std::vector<int> distance2center (Y*Z, 0);
			
			_image.LengthsPathToStart( distance2center, _i);
			
			GetPathGraph( distance2center, _goal, _i);
		}


		void GetPathGraph( std::vector<int>& _lengths, cv::Point2i& _goal, int& _i)
		{
			nodes[_i].clear();
			Node node;
			node.pos = _goal;
			cv::Point2i p = _goal;			
			node.parent = 0;
			nodes[_i].push_back(node);	
			for( int i = 0; i < _lengths[ _goal.x *Z + _goal.y] ; i++ )
			{

				if( _lengths[ p.x *Z + p.y] <= 1) return;
				for( auto& b: boundary_ )
					ChangeIfCloserNode( p, b, _lengths, _i);
			}	
		}



		void ChangeIfCloserNode( cv::Point2i& _node_start, const cv::Point2i& _move, std::vector<int>& _lengths, int& _i)
		{
			if( isNodePathCloseToStart(_node_start, _node_start + _move, _lengths))
			{
				_node_start += _move;
				Node node;
				node.pos = _node_start;
				node.parent = nodes[_i].size()-1;
				nodes[_i].push_back(node);
			}	
		}


		bool isNodePathCloseToStart( const cv::Point2i& _p_c, const cv::Point2i& _p_n, std::vector<int>& _lengths)
		{
			return (_lengths[ _p_c.x *Z + _p_c.y] - 1) == _lengths[ _p_n.x *Z + _p_n.y];
		}	
		
		void SimplifyGraph( int _i )
		{			
			for(int n = nodes[_i].size()-1; n>0; n--)
			{
				if(!nodes[_i][n].parent)
				{
					nodes[_i].erase(nodes[_i].begin()+n);
					for( int j = n; j < static_cast<int>( nodes[_i].size() ); j++)
					{
						if( nodes[_i][j].parent && *nodes[_i][j].parent >= n )
						{
							*nodes[_i][j].parent -= 1;
						}	
					}
				}
			}
		}
		
		void UpdateConnections( GeneralizedVector3D<X, Y, Z>& _image, int _i )
		{
			for( size_t j = 0; j < nodes[_i].size(); j++)
			{
				nodes[_i][j].data_comp.data.connections.clear();
				for( size_t k = 0; k < nodes[_i].size(); k++)
				{
					if( j !=k && Utils::Measures::ComputeDist( nodes[_i][j].pos, nodes[_i][k].pos) < RADIUS)
					{
						if( !_image.isColliding( nodes[_i][j].pos, nodes[_i][k].pos, _i))
						{
							nodes[_i][j].data_comp.data.connections.push_back(k);
						}
					}
				}
			}
		}
		
		void Save( std::string dir, int _i )
		{
			std::ofstream file;
			file.open(dir + "/nodes.n");  
			for( size_t j = 0; j < nodes[_i].size(); j++)
			{
				auto data = nodes[_i][j].GetNodeInfo();

				file.write(reinterpret_cast<const char*>(&data.v_size),sizeof(data.v_size));
				file.write(reinterpret_cast<const char*>(&data.data.pos_x),sizeof(data.data.pos_x));
				file.write(reinterpret_cast<const char*>(&data.data.pos_y),sizeof(data.data.pos_y));
				file.write(reinterpret_cast<const char*>(&data.data.parent),sizeof(data.data.parent));
				for(auto a: data.data.connections)
					file.write(reinterpret_cast<const char*>(&a),sizeof(a));
			}			
			file.close();
		}
		
		
		void Load( std::string& dir, int _i )
		{	
			try
			{
				nodes[_i].clear();

				std::ifstream file(dir + "/nodes.n", std::ios::binary);
				
				int j=0;
				nodes[_i].emplace_back();
				
				bool end_f = true;

				end_f = (file.read((char*)&nodes[_i][j].data_comp.v_size,sizeof(size_t)))?true:false;	
				
				while( end_f)
				{
					
					nodes[_i][j].data_comp.data.connections.resize((nodes[_i][j].data_comp.v_size-3*sizeof(uint16_t))/sizeof(uint16_t),0);
		
					end_f &= file.read((char*)&nodes[_i][j].data_comp.data,3*sizeof(uint16_t))?true:false;				
					end_f &= file.read((char*)nodes[_i][j].data_comp.data.connections.data(),(nodes[_i][j].data_comp.v_size-3*sizeof(uint16_t)))?true:false;
					
					nodes[_i][j].SetNodeInfo();
												
					nodes[_i].emplace_back();

					++j;	

					end_f &= file.read((char*)&nodes[_i][j].data_comp.v_size,sizeof(size_t))?true:false;			
				}	
				
				file.close();

				nodes[_i].erase( nodes[_i].end());
			}
			catch( std::exception& e)
			{ 
				Utils::File::RemoveDir( dir );
				std::cout<<"Error Loading --------------------------------------"<<std::endl; 
			}	
			
		}
	
	
	private:
		float ComputeCost( const Node& _n_start, const Node& _n_end, const float& _dist)	
		{ 
			return _n_start.acc_cost + _dist*(_n_start.pos_cost + _n_end.pos_cost);
		}
		
	
		
		
};

#endif
