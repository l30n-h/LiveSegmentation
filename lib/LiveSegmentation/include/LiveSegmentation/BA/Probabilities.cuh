#ifndef LIVESEGMENTATION_PROBABILITIES_H
#define LIVESEGMENTATION_PROBABILITIES_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <VoxelHasher/Core/HostDeviceObject.cuh>

namespace ls {

using LPPair = thrust::pair<unsigned int, float>;
#ifdef __CUDA_ARCH__
template <typename T> using Vector = thrust::device_vector< T >;
#else
template <typename T> using Vector = thrust::host_vector<   T >;
#endif

class Probabilities
{
	public:

		__host__ __device__
		Probabilities();

		__host__ __device__
		Probabilities(const Probabilities& p);
		
		__device__ __host__
		void update(const Probabilities& p);

		__device__ __host__
		LPPair getMax();

	private:
		Vector<LPPair> probabilities;
};

}
#endif // LIVESEGMENTATION_PROBABILITIES_H
