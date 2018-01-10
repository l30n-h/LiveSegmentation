#include "LiveSegmentation/BA/Classifier.hpp"

#include <utility>

#include <caffe/caffe.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace ls {

class Classifier::Impl
{
	public:
        Impl(const std::string& model_file,
                       const std::string& trained_file,
                       const cv::Scalar mean,
                       const std::vector<std::string> labels,
                       const bool force_cpu);
        ~Impl();

		void updateThreadSpecificSettings();
		bool usesGPU();
		std::vector<cv::Mat> Predict(const cv::Mat& img);		
		std::pair<cv::Mat, cv::Mat> Classify(const std::vector<cv::Mat> predictions);
		void WrapLayer(caffe::Blob<float>* layer, std::vector<cv::Mat>* channels);
		void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

		caffe::shared_ptr<caffe::Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
		cv::Scalar mean_;
		std::vector<std::string> labels_;
		bool force_cpu_ = false;
};

Classifier::Classifier(const Classifier& op)
: impl(std::make_unique<Impl>(*op.impl))
{}

Classifier& Classifier::operator=(const Classifier& op) {
    if (this != &op) {
        impl.reset(new Impl(*op.impl));
    }
    return *this;
}


Classifier::Classifier(const std::string& model_file,
                       const std::string& trained_file,
                       const cv::Scalar mean,
                       const std::vector<std::string> labels,
                       const bool force_cpu) 
:impl(std::make_unique<Impl>(model_file, trained_file, mean, labels, force_cpu)){
}

Classifier::~Classifier()
{
	
}

std::pair<cv::Mat, cv::Mat> Classifier::Classify(const std::vector<cv::Mat> predictions) {
	return impl->Classify(predictions);
}

void Classifier::updateThreadSpecificSettings(){
	impl->updateThreadSpecificSettings();
}

bool Classifier::usesGPU(){
	return impl->usesGPU();
}

std::vector<cv::Mat> Classifier::Predict(const cv::Mat& img) {
	return impl->Predict(img);
}


Classifier::Impl::Impl(const std::string& model_file,
                       		const std::string& trained_file,
                       		const cv::Scalar mean,
                       		const std::vector<std::string> labels,
                       		const bool force_cpu){
	/* Load the network. */
	net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	if(net_->num_inputs() != 1) throw "Network should have exactly one input.";
	if(net_->num_outputs() != 1) throw "Network should have exactly one output.";

	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	if(num_channels_ != 3) throw "Input layer should have 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	mean_ = mean;

	//Blob<float>* output_layer = net_->output_blobs()[0];
	//if(labels.size() != output_layer->channels()) throw "Number of labels is different from the output layer dimension.";
	labels_ = labels;

	updateThreadSpecificSettings();
	force_cpu_ = force_cpu;
}

Classifier::Impl::~Impl()
{

}

void Classifier::Impl::updateThreadSpecificSettings(){
#ifdef CPU_ONLY
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
	caffe::Caffe::set_mode(force_cpu_ ? caffe::Caffe::CPU : caffe::Caffe::GPU);
#endif
}

bool Classifier::Impl::usesGPU(){
	return caffe::Caffe::mode() == caffe::Caffe::GPU;
}

std::vector<cv::Mat> Classifier::Impl::Predict(const cv::Mat& img) {
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
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

std::pair<cv::Mat, cv::Mat> Classifier::Impl::Classify(const std::vector<cv::Mat> predictions) {
	if(predictions.size()<=0) return std::pair<cv::Mat, cv::Mat>();
	size_t rows = predictions[0].rows;
	size_t cols = predictions[0].cols;
	cv::Mat maxClass = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::Mat maxProb = cv::Mat::zeros(rows, cols, CV_8UC1);
	for (size_t i = 0; i < predictions.size(); ++i){
		cv::Mat oi = predictions[i];
		if(oi.rows>=0 && oi.cols>=0 && ((unsigned)oi.rows)==rows && ((unsigned)oi.cols)==cols){
			for(size_t row = 0; row < rows; ++row){
				for(size_t col = 0; col < cols; ++col){
					unsigned char value = (unsigned char)(((float)oi.at<float>(row,col))*255);
					if (value > (unsigned char)maxProb.at<unsigned char>(row,col)){
						maxProb.at<unsigned char>(row,col) = value;
						maxClass.at<unsigned char>(row,col) = (unsigned char)(i+1);
					}
				}
			}
		}
	}
	return std::make_pair(maxClass, maxProb);
}

/* Wrap the layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the layer. */
void Classifier::Impl::WrapLayer(caffe::Blob<float>* layer, std::vector<cv::Mat>* channels) {
	int width = layer->width();
	int height = layer->height();
	float* data = layer->mutable_cpu_data();
	for (int i = 0; i < layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, data);
		channels->push_back(channel);
		data += width * height;
	}
}

void Classifier::Impl::Preprocess(const cv::Mat& img,
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

  if(reinterpret_cast<float*>(input_channels->at(0).data) != net_->input_blobs()[0]->cpu_data()){
    throw "Input channels are not wrapping the input layer of the network";
  }
    
}

}
