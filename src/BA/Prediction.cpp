#include "Prediction.hpp"

#include <thrust/reduce.h>

//namespace ba {

__host__ __device__
Prediction::Prediction()
{

}

__host__ __device__
Prediction::Prediction(const Prediction& p)
	: prediction(p) 
{

}

calcProbabilty(float lastProb, float newProb){
	float alpha = 0.8f;
	return oldProb + alpha*(prob-oldProb);
}

__device__ __host__
void update(const Prediction& p)
{
	Vector::iterator end1 = prediction.end();
    Vector::iterator i2 = p.begin();
	Vector::iterator end2 = p.end();
	for(; i1 != end1; ++i1) {
		LPPair p1 = (*l1);
		while(i2 != end2) {
			LPPair p2 = (*i2);
			if(p1.label<=p2.label) break;
			prediction.append(p2);
			++i2
		}
		prob2 = 0;
		if(i2 != end2 && (*i2).first==p1.first){
			prob2 = (*i2).second
		}
		p1.second = calcProbabilty(p1.second, prob2);
		
	}
	for(; i2 != end2; ++i2) {
		prediction.append((*i2));
	}
	if(prediction.end() != end1){
		//sort
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
	return thrust::reduce(prediction.begin(), prediction.end(), LPPair(0,0), maxByProbabilty());
}

//} // namespace
