#include "LiveSegmentation/BA/ClassifierQueue.hpp"

#include <functional>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <list>
#include <utility>

namespace ls {

class ClassifierQueue::Impl
{
	public:
        Impl(Classifier& classifier);
        ~Impl();

		void start();
		void stop();
		void add(Image image, Consumer consumer);
		void run();

		size_t limit = 1;
		bool isRunning = false;
		std::mutex mutex;
		std::mutex conditionMutex;
		std::condition_variable conditionVariable;
		std::unique_ptr<std::thread> workerThread;

		std::list< std::pair<Image, Consumer> > list;
		Classifier classifier;
};
/*
ClassifierQueue::ClassifierQueue(const ClassifierQueue& op)
: impl(new Impl(*op.impl))
{}

ClassifierQueue& ClassifierQueue::operator=(const ClassifierQueue& op) {
    if (this != &op) {
        impl.reset(new Impl(*op.impl));
    }
    return *this;
}*/


ClassifierQueue::ClassifierQueue(Classifier& classifier)
:impl(std::make_unique<Impl>(classifier))
{

}


ClassifierQueue::~ClassifierQueue()
{
	
}

Classifier
ClassifierQueue::getClassifier()
{
	return impl->classifier;
}

void
ClassifierQueue::start()
{
	impl->start();
}


void
ClassifierQueue::stop()
{
	impl->stop();
}

void
ClassifierQueue::add(Image image, Consumer consumer)
{
	impl->add(image, consumer);
}


ClassifierQueue::Impl::Impl(Classifier& classifier)
:classifier(classifier)
{

}

ClassifierQueue::Impl::~Impl()
{
	stop();
}

void
ClassifierQueue::Impl::start()
{
	if(!isRunning){
		isRunning = true;
		workerThread = std::make_unique<std::thread>(&ClassifierQueue::Impl::run, this);
	}
}


void
ClassifierQueue::Impl::stop()
{
	if(isRunning){
		isRunning = false;
		conditionVariable.notify_all();
		workerThread->join();
	}
}

void
ClassifierQueue::Impl::add(Image image, Consumer consumer)
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
	}
	conditionVariable.notify_one();
	if(isDeleted) {
		pair.second(std::vector<Image>());
	}
}

void
ClassifierQueue::Impl::run()
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

}
