#ifndef LIVESEGMENTATION_PROBABILITIES_H
#define LIVESEGMENTATION_PROBABILITIES_H

#include <vector>
#include <memory>
#include <utility>
#include <opencv2/core/core.hpp>

namespace ls {

using LPPair = std::pair<unsigned int, float>;

class Probabilities
{

	public:

		Probabilities();

		~Probabilities();
		
		void update(const std::vector<cv::Mat>& predictions, unsigned int x, unsigned int y);

		LPPair getMax();

	
	private:
		class Impl;
		std::shared_ptr<Impl> impl;
};

}
#endif // LIVESEGMENTATION_PROBABILITIES_H
