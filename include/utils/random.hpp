#ifndef RANDOM_H
#define RANDOM_H

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <random>
#include <chrono>
#include <limits>


namespace Utils
{
namespace Random
{	
    template<typename T>
    using UniformDistribution = std::conditional_t<std::is_integral<T>::value, std::uniform_int_distribution<T>,
                                std::conditional_t<std::is_floating_point<T>::value, std::uniform_real_distribution<T>,
                                void>>;

	
	std::mt19937 getGenerator(void)
	{
		std::mt19937           generator(std::random_device{}());
		return generator;
	}
	
	template<typename T>
	UniformDistribution<T> getUniformDistribution(auto _min = std::numeric_limits<T>::min(), auto _max = std::numeric_limits<T>::max())
	{
        UniformDistribution<T> distribution(_min, _max);        
        return distribution;
	}	

    template<typename T>
    auto getRandomNumber(auto _min = std::numeric_limits<T>::min(), auto _max = std::numeric_limits<T>::max())
    {
        std::mt19937           generator(std::random_device{}());
        UniformDistribution<T> distribution(_min, _max);

        return distribution(generator);
    }
   

    template<typename T>
    struct isDuration : std::false_type {};

    template<typename Rep, typename Period>
    struct isDuration<std::chrono::duration<Rep, Period>> : std::true_type {};

    template<typename Duration>
    auto getRandomDuration(Duration _min = Duration::min(), Duration _max = Duration::max())
    {
        static_assert(isDuration<Duration>::value, "Template argument must be a std::chrono::duration");
        return Duration(getRandomNumber<typename Duration::rep>(_min.count(), _max.count()));
    }

    template<typename Iterator>
    auto getRandomIterator(Iterator _first, Iterator _end)
    {
        auto range_distance = std::distance(_first, _end);
        if(range_distance > 0) { range_distance--; }
        std::advance(_first, getRandomNumber<typename std::iterator_traits<Iterator>::difference_type>(0, range_distance));

        return _first;
    }
    
	void AddRandomness(const tensorflow::Tensor& t)
	{
		auto t_float = t.flat<float>();
		for(int i = 0; i < t_float(0); i++)
		{
			getRandomNumber<int>(0,100000);
		}
	}
}
}

#endif
