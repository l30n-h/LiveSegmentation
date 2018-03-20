#include "LiveSegmentation/BA/Probabilities.hpp"

#include <algorithm>
#include <vector>

namespace ls {

template <typename T> using Vector = std::vector<   T >;

class Probabilities::Impl
{
	// public:
	// 	Impl();
	// 	~Impl();

	public:
		Vector<LPPair> probabilities;
			
};

Probabilities::Probabilities()
:impl(std::make_shared<Impl>())
{

}

Probabilities::~Probabilities()
{

}

static float calcProbabilty(float lastProb, float newProb){
	float alpha = 0.5f;
	return lastProb + alpha*(newProb-lastProb);
}


void Probabilities::update(const std::vector<cv::Mat>& predictions, unsigned int x, unsigned int y)
{
	if(impl->probabilities.size()==0){
		impl->probabilities.reserve(predictions.size());
		for(unsigned int i=0;i<predictions.size();i++){
			impl->probabilities.push_back(LPPair(i+1, predictions[i].at<float>(y, x)));
		}
	}else{
		for(unsigned int i=0;i<predictions.size();i++){
			impl->probabilities[i].second = calcProbabilty(impl->probabilities[i].second, predictions[i].at<float>(y, x));
		}
	}
}

static bool compareMax(LPPair a, LPPair b)
{
    return a.second < b.second;
}

LPPair Probabilities::getMax()
{
	if(impl->probabilities.size()>0){
		return *std::max_element(impl->probabilities.begin(), impl->probabilities.end(), compareMax);
	}
	return LPPair(0,0);
}

}
