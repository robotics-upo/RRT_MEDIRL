#ifndef FILE_ACTIONS_H
#define FILE_ACTIONS_H

#include <fstream>
#include <experimental/filesystem>

namespace Utils
{
namespace File
{	
	void Write(std::string filename, std::string content)
	{
		std::ofstream myfile;
		myfile.open(filename.c_str());               
		myfile << content;
		myfile.close();
	}
	
	bool Compare(std::string filename, std::string content)
	{
		std::ifstream myfile( filename.c_str() ) ;
		std::string file_content;
		myfile >> file_content;
		myfile.close();
		return content.compare(file_content);
	}	

	bool CreateDir( std::string dir )
	{
		if (!std::experimental::filesystem::exists(dir))
		{
			std::experimental::filesystem::create_directories(dir);
			return true;
		}
		return false;
	}
	
	bool RemoveDir( std::string dir )
	{
		try
		{
<<<<<<< HEAD
			try
			{
				std::experimental::filesystem::remove_all(dir);
				std::cout<<"Deleting "<<dir<<std::endl;
			}
			catch(int e)
			{}
			
			return true;
=======
			if (std::experimental::filesystem::exists(dir))
			{
				std::experimental::filesystem::remove_all(dir);
				std::cout<<"Deleting "<<dir<<std::endl;
				
				return true;
			}
>>>>>>> b29099f8bab1f37dc880d5db664e5507d08b77b7
		}
		catch( int i)
		{}
		return false;
			
	}	
	
}
}

#endif
