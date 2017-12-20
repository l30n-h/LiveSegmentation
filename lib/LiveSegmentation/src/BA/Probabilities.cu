#include "LiveSegmentation/BA/Probabilities.hpp"

#include <thrust/reduce.h>
#include <thrust/sort.h>

namespace ls {

__host__ __device__
Probabilities::Probabilities()
{

}

__host__ __device__
Probabilities::Probabilities(const Probabilities& p)
	: probabilities(p) 
{

}

calcProbabilty(float lastProb, float newProb){
	float alpha = 0.8f;
	return oldProb + alpha*(prob-oldProb);
}

struct sortByLabel {
  __host__ __device__
  bool operator()(const LPPair& p1, const LPPair& p2) {
      return p1.first < p2.first;
  }
};

__device__ __host__
void update(const Probabilities& p)
{
	Vector::iterator end1 = probabilities.end();
	Vector::iterator i2 = p.begin();
	Vector::iterator end2 = p.end();
	for(; i1 != end1; ++i1) {
		LPPair p1 = (*l1);
		while(i2 != end2) {
			LPPair p2 = (*i2);
			if(p1.label<=p2.label) break;
			probabilities.append(p2);
			++i2
		}
		prob2 = 0;
		if(i2 != end2 && (*i2).first==p1.first){
			prob2 = (*i2).second
		}
		p1.second = calcProbabilty(p1.second, prob2);
		
	}
	for(; i2 != end2; ++i2) {
		probabilities.append((*i2));
	}
	if(probabilities.end() != end1){
		thrust::sort(probabilities.begin(), probabilities.end(), sortByLabel());//TODO sort in O(n)
	}
}

struct maxByProbabilty {
     __host__ __device__
        LPPair operator()(const LPPair& p1, const LPPair& p2) const {
        return p1.second>=p2.second?p1:p2;
    }
};

__device__ __host__
LPPair getMax()
{
	return thrust::reduce(probabilities.begin(), probabilities.end(), LPPair(0,0), maxByProbabilty());
}

}
