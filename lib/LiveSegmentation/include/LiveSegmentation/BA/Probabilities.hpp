#ifndef LIVESEGMENTATION_PROBABILITIES_H
#define LIVESEGMENTATION_PROBABILITIES_H

#include <vector>
#include <utility>
#include <opencv2/core/core.hpp>

namespace ls {

using LPPair = std::pair<unsigned int, float>;
template <typename T> using Vector = std::vector<   T >;

class Probabilities
{
	public:

		Probabilities();
		
		void update(std::vector<cv::Mat>& predictions, unsigned int x, unsigned int y);

		LPPair getMax();

	private:
		Vector<LPPair> probabilities;
};

}
#endif // LIVESEGMENTATION_PROBABILITIES_H
