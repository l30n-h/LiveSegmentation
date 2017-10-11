#ifndef BA_CLASSIFIER_H
#define BA_CLASSIFIER_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>

class Classifier {
	public:
		Classifier(const std::string& model_file,
					const std::string& trained_file,
					const cv::Scalar mean,
					const std::vector<std::string> labels);

		std::vector<cv::Mat> Predict(const cv::Mat& img);		

		std::pair<cv::Mat, cv::Mat> Classify(const std::vector<cv::Mat> predictions);

	private:
		void WrapLayer(caffe::Blob<float>* layer, std::vector<cv::Mat>* channels);

		void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

	private:
		caffe::shared_ptr<caffe::Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
		cv::Scalar mean_;
		std::vector<std::string> labels_;
};

#endif // BA_CLASSIFIER_H

