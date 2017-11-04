#ifndef BA_Prediction_H
#define BA_Prediction_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <VoxelHasher/Core/HostDeviceObject.cuh>

using LPPair = thrust::pair<unsigned int, float>;
#ifdef __CUDA_ARCH__
template <typename T> using Vector = thrust::device_vector< T >;
#else
template <typename T> using Vector = thrust::host_vector<   T >;
#endif

class Prediction
{
	public:

		__host__ __device__
		Prediction();

		__host__ __device__
		Prediction(const Prediction& p);
		
		__device__ __host__
		void update(const Prediction& p);

		__device__ __host__
		LPPair getMax();

	private:
		Vector<LPPair> prediction;
};

#endif // BA_Prediction_H
