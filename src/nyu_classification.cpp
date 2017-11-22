#include "BA/Classifier.hpp"
#include "BA/ClassifierQueue.hpp"

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <thread>
#include <chrono>

#ifdef USE_OPENCV

/* Load the mean file in binaryproto format. */
static cv::Scalar getMeanFromFile(const std::string& mean_file) {
	caffe::BlobProto blob_proto;
	caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	caffe::Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	size_t num_channels = mean_blob.channels();

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value */
	return cv::mean(mean);
}

static std::vector<std::string> getLabelsFromFile(const std::string& label_file){
	/* Load labels. */
	std::ifstream labelStream(label_file.c_str());
	CHECK(labelStream) << "Unable to open labels file " << label_file;

	std::vector<std::string> labels;
	std::string line;
	while (std::getline(labelStream, line))
		labels.push_back(std::string(line));
	return labels;
}

static void printWithTime(std::string msg){
	time_t t = time(0);
	struct tm * now = localtime( & t );
	std::cout << "[";
	if(now->tm_hour<10) std::cout << "0";
	std::cout << now->tm_hour << ":";
	if(now->tm_min<10) std::cout << "0";
	std::cout << now->tm_min << ":";
	if(now->tm_sec<10) std::cout << "0";
	std::cout << now->tm_sec << "]\t";
	std::cout << msg << std::endl;
}

int main(int argc, char** argv) {
	if (argc < 7) {
		std::cerr << "Usage: " << argv[0]
					<< " <default|force_cpu>"
					<< " deploy.prototxt network.caffemodel"
					<< " mean_b mean_g mean_r img1.jpg [img2.jpg ...]" << std::endl;
		return 1;
	}

	::google::InitGoogleLogging(argv[0]);
	bool force_cpu = "force_cpu" == std::string(argv[1]);
	std::string model_file   = argv[2];
	std::string trained_file = argv[3];
	float mean_b = (float)std::atof(argv[4]);
	float mean_g = (float)std::atof(argv[5]);
	float mean_r = (float)std::atof(argv[6]);
	Classifier classifier(model_file, trained_file, cv::Scalar(mean_b, mean_g, mean_r), std::vector<std::string>(), force_cpu);
	printWithTime(classifier.usesGPU()?"Runs on GPU":"Runs on CPU");
		
	ClassifierQueue classifierQueue(classifier);
	classifierQueue.start();
	
	printWithTime("Prediction started");
	int isDone = argc-7;
	for(int i=7;i<argc;i++){
		std::string file = argv[i];
		printWithTime("Read image "+file);
		cv::Mat img = cv::imread(file, -1);
		CHECK(!img.empty()) << "Unable to decode image " << file;
	
		
		classifierQueue.add(img, [&classifierQueue, &isDone, &argc, i](std::vector<cv::Mat> predictions) mutable {
			std::string num = std::to_string(i-6);
			if(predictions.empty()){
				printWithTime("Prediction skipped for "+num);
			} else{
				printWithTime("Prediction done for "+num);
				printWithTime("Classification started");
				std::pair<cv::Mat, cv::Mat> result = classifierQueue.getClassifier().Classify(predictions);
				printWithTime("Classification done");
				cv::imwrite( "./Classes"+num+".png", result.first );
				cv::imwrite( "./Probs"+num+".png", result.second );
			}
			isDone--;
		});
	}
	while(isDone>0){
		printWithTime("Still predicting");
		std::this_thread::sleep_for(std::chrono::seconds(10));
	}
	classifierQueue.stop();
	printWithTime("Done");

	/*
	Classifier classifier(model_file, trained_file, cv::Scalar(mean_b, mean_g, mean_r), std::vector<std::string>());
	printWithTime("Prediction started");
	for(int i=6;i<argc;i++){		
		std::string file = argv[i];
		printWithTime("Read image "+file);
		cv::Mat img = cv::imread(file, -1);
		CHECK(!img.empty()) << "Unable to decode image " << file;

		
		std::vector<cv::Mat> predictions = classifier.Predict(img);
		printWithTime("Prediction done");
		printWithTime("Classification started");
		std::pair<cv::Mat, cv::Mat> result = classifier.Classify(predictions);
		printWithTime("Classification done");
		cv::imwrite( "./Classes.png", result.first );
		cv::imwrite( "./Probs.png", result.second );
		
	}
	printWithTime("Done");*/
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV

