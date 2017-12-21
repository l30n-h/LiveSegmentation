#include <LiveSegmentation/BA/Classifier.hpp>
#include <LiveSegmentation/BA/ClassifierQueue.hpp>

#include <iostream>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <thread>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace ls;

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

	bool force_cpu = "force_cpu" == std::string(argv[1]);
	std::string model_file   = argv[2];
	std::string trained_file = argv[3];
	float mean_b = (float)std::atof(argv[4]);
	float mean_g = (float)std::atof(argv[5]);
	float mean_r = (float)std::atof(argv[6]);
	Classifier classifier(model_file, trained_file, cv::Scalar(mean_b, mean_g, mean_r), std::vector<std::string>(), force_cpu);
	printWithTime(classifier.usesGPU()?"Runs on GPU":"Runs on CPU");
	
	printWithTime("Prediction started");
	for(int i=7;i<argc;i++){		
		std::string file = argv[i];
		printWithTime("Read image "+file);
		cv::Mat img = cv::imread(file, -1);
		if(img.empty()){
			std::cerr << "Unable to decode image " << file;
			continue;
		}
		
		std::vector<cv::Mat> predictions = classifier.Predict(img);
		printWithTime("Prediction done");
		printWithTime("Classification started");
		std::pair<cv::Mat, cv::Mat> result = classifier.Classify(predictions);
		printWithTime("Classification done");
		cv::imwrite( "./ClassesNQ.png", result.first );
		cv::imwrite( "./ProbsNQ.png", result.second );
		
	}
	printWithTime("Done");


	ClassifierQueue classifierQueue(classifier);
	classifierQueue.start();
	
	printWithTime("Prediction started");
	int isDone = argc-7;
	for(int i=7;i<argc;i++){
		std::string file = argv[i];
		printWithTime("Read image "+file);
		cv::Mat img = cv::imread(file, -1);
		if(img.empty()){
			std::cerr << "Unable to decode image " << file;
			continue;
		}
		
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
}

