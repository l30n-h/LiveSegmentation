#include "LiveSegmentation/BA/Probabilities.hpp"

#include <algorithm>

namespace ls {

Probabilities::Probabilities()
{

}

static float calcProbabilty(float lastProb, float newProb){
	float alpha = 0.2f;
	return lastProb + alpha*(newProb-lastProb);
}


void Probabilities::update(std::vector<cv::Mat>& predictions, unsigned int x, unsigned int y)
{
	if(probabilities.size()==0){
		for(unsigned int i=0;i<predictions.size();i++){
			probabilities.push_back(LPPair(i+1, predictions[i].at<float>(y, x)));
		}
	}else{
		for(unsigned int i=0;i<predictions.size();i++){
			probabilities[i].second = calcProbabilty(probabilities[i].second, predictions[i].at<float>(y, x));
		}
	}
}

static bool compareMax(LPPair a, LPPair b)
{
    return a.second < b.second;
}

LPPair Probabilities::getMax()
{
	if(probabilities.size()>0){
		return *std::max_element(probabilities.begin(), probabilities.end(), compareMax);
	}
	return LPPair(0,0);
}

}
