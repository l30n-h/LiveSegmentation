#ifndef LIVESEGMENTATION_CLASSIFIER_H
#define LIVESEGMENTATION_CLASSIFIER_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace ls {

class Classifier {
	public:
		Classifier(const std::string& model_file,
					const std::string& trained_file,
					const cv::Scalar mean,
					const std::vector<std::string> labels,
                    const bool force_cpu=false);
		~Classifier();

		Classifier(Classifier && op) noexcept;
    	Classifier& operator=(Classifier && op) noexcept;

    	Classifier(const Classifier& op);
    	Classifier& operator=(const Classifier& op);


		bool usesGPU();
		std::vector<cv::Mat> Predict(const cv::Mat& img);		
		std::pair<cv::Mat, cv::Mat> Classify(const std::vector<cv::Mat> predictions);

	private:
		class Impl;
		std::unique_ptr<Impl> impl;
};

}
#endif // LIVESEGMENTATION_CLASSIFIER_H

