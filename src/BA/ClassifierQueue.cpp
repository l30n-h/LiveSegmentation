#include "ClassifierQueue.hpp"

//namespace ba {

ClassifierQueue::ClassifierQueue(Classifier& classifier): 
classifier(classifier)
{
}


ClassifierQueue::~ClassifierQueue()
{
	stop();
}

Classifier
ClassifierQueue::getClassifier()
{
	return classifier;
}

void
ClassifierQueue::start()
{
	isRunning = true;
	workerThread = std::make_unique<std::thread>(&ClassifierQueue::ClassifierQueue::run, this);
}


void
ClassifierQueue::stop()
{
	isRunning = false;
	conditionVariable.notify_all();
	workerThread->join();
}


void
ClassifierQueue::run()
{
	while(isRunning) {
		std::pair<Image, Consumer> pair;
		bool isEmpty = true;
		{
			std::lock_guard<std::mutex> guard(mutex);
			if (!list.empty()) {
				isEmpty = false;
				pair = list.front();
				list.pop_front();
			}
		}
		if(isEmpty) {
			std::unique_lock<std::mutex> conditionLock(conditionMutex);
			conditionVariable.wait(conditionLock);
		}else {
			std::vector<Image> predictions = classifier.Predict(pair.first);
			pair.second(predictions);
		}
	}
}

void
ClassifierQueue::add(Image image, Consumer consumer)
{
	std::pair<Image, Consumer> pair;
	bool isDeleted = false;	
	{
		std::lock_guard<std::mutex> guard(mutex);
		if(list.size()==limit) {
			isDeleted = true;
			pair = list.back();
			list.pop_back();
		}
		list.push_back(std::make_pair(image, consumer));
		conditionVariable.notify_one();
	}
	if(isDeleted) {
		pair.second(std::vector<Image>());
	}
}

//} // namespace
