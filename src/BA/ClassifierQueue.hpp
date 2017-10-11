#ifndef BA_CLASSIFIERQUEUE_H
#define BA_CLASSIFIERQUEUE_H

#include <memory>
#include <functional>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <list>
#include <utility>

#include <opencv2/core/core.hpp>
#include "Classifier.hpp"

typedef cv::Mat Image;
typedef std::function<void(std::vector<Image>)> Consumer;

class ClassifierQueue
{
	public:

		ClassifierQueue(Classifier& classifier);
		~ClassifierQueue();
		
		Classifier getClassifier();
		void start();
		void stop();
		void add(Image image, Consumer consumer);
	private:
		void run();

	private:
		size_t limit = 1;
		bool isRunning;
		std::mutex mutex;
		std::mutex conditionMutex;
		std::condition_variable conditionVariable;
		std::unique_ptr<std::thread> workerThread;

		std::list< std::pair<Image, Consumer> > list;
		Classifier classifier;
};

#endif // BA_CLASSIFIERQUEUE_H
