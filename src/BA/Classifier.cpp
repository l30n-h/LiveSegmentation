#include "Classifier.hpp"


#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;  // NOLINT(build/namespaces)

//namespace ba {

Classifier::Classifier(const std::string& model_file,
                       const std::string& trained_file,
                       const cv::Scalar mean,
                       const std::vector<std::string> labels) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3)
		<< "Input layer should have 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	mean_ = mean;

	//Blob<float>* output_layer = net_->output_blobs()[0];
	//CHECK_EQ(labels.size(), output_layer->channels())
	//	<< "Number of labels is different from the output layer dimension.";
	labels_ = labels;
}

std::pair<cv::Mat, cv::Mat> Classifier::Classify(const std::vector<cv::Mat> predictions) {
	if(predictions.size()<=0) return std::pair<cv::Mat, cv::Mat>();
	size_t rows = predictions[0].rows;
	size_t cols = predictions[0].cols;
	cv::Mat maxClass (rows, cols, CV_8UC1);
	cv::Mat maxProb (rows, cols, CV_8UC1);
	for (size_t i = 0; i < predictions.size(); ++i){
		cv::Mat oi = predictions[i];
		if(oi.rows>=0 && oi.cols>=0 && ((unsigned)oi.rows)==rows && ((unsigned)oi.cols)==cols){
			for(size_t row = 0; row < rows; ++row){
				for(size_t col = 0; col < cols; ++col){
					unsigned char value = (unsigned char)(((float)oi.at<float>(row,col))*255);
					if (value > (unsigned char)maxProb.at<unsigned char>(row,col)){
						maxProb.at<unsigned char>(row,col) = value;
						maxClass.at<unsigned char>(row,col) = (unsigned char)i;
					}
				}
			}
		}
	}
	return std::make_pair(maxClass, maxProb);
}

std::vector<cv::Mat> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapLayer(net_->input_blobs()[0], &input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();	

	std::vector<cv::Mat> output_channels;
	WrapLayer(net_->output_blobs()[0], &output_channels);
	return output_channels;
}

/* Wrap the layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the layer. */
void Classifier::WrapLayer(Blob<float>* layer, std::vector<cv::Mat>* channels) {
	int width = layer->width();
	int height = layer->height();
	float* data = layer->mutable_cpu_data();
	for (int i = 0; i < layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, data);
		channels->push_back(channel);
		data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

//} // namespace
