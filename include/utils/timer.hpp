#ifndef TIMER_H
#define TIMER_H

#include <ctime>
#include <ratio>
#include <chrono>

namespace Utils
{
namespace Timer
{
	class Timer
	{
		private:
			std::chrono::high_resolution_clock::time_point t;
			double t_start {  0.0  };
			bool timer_on  { false };
			
		public:
			Timer(){ t = std::chrono::high_resolution_clock::now(); };
			~Timer() = default;
		
			void   tic()
			{
				t = std::chrono::high_resolution_clock::now(); 
				t_start = 0.0; 
				timer_on = true; 
			} 
			double toc()
			{ 
				return t_start + (timer_on ? std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t).count() : 0.0); 
			}
			void t_pause()
			{ 
				if(timer_on) 
				{ 
					t_start += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t).count();  
					timer_on = false; 
				}
			}
			void   t_continue()
			{ 
				if( !timer_on )
				t = std::chrono::high_resolution_clock::now(); 
				timer_on = true; 
			}
	};	
}
}



#endif
