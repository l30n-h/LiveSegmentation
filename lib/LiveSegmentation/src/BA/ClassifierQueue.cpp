#include "LiveSegmentation/BA/ClassifierQueue.hpp"

#include <functional>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <utility>

namespace ls {

class ClassifierQueue::Impl
{
	public:
        Impl(Classifier& classifier);
        ~Impl();

		void start();
		void stop();
		void setLimit(size_t limit, bool overwrite);
		void add(Image image, Consumer consumer);
		void run();

		size_t limit = 1;
		bool overwrite = true;
		bool isRunning = false;
		std::mutex mutex;
		std::mutex conditionMutex;
		std::condition_variable conditionVariable;
		std::unique_ptr<std::thread> workerThread;

		std::deque< std::pair<Image, Consumer> > queue;
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
ClassifierQueue::setLimit(size_t limit, bool overwrite)
{
	impl->setLimit(limit, overwrite);
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
ClassifierQueue::Impl::setLimit(size_t pLimit, bool pOverwrite)
{
	{	
		std::lock_guard<std::mutex> guard(mutex);
		if(pLimit<limit){
			std::pair<Image, Consumer> pair;
			while(queue.size()>pLimit){
				pair = queue.back();
				queue.pop_back();
				pair.second(std::vector<Image>());
			}
		}
		limit = pLimit;
		overwrite=pOverwrite;
	}
}

void
ClassifierQueue::Impl::add(Image image, Consumer consumer)
{
	Consumer deletedConsumer;
	bool isDeleted = false;
	bool isFull = false;
	{
		std::lock_guard<std::mutex> guard(mutex);
		isFull=queue.size()==limit;
		if(isFull) {
			if(overwrite){
				deletedConsumer = queue.back().second;
				queue.pop_back();
				isFull = false;
			}else {
				deletedConsumer = consumer;
			}
			isDeleted = true;
		}
		if(!isFull) queue.push_back(std::make_pair(image, consumer));
	}
	if(!isFull) conditionVariable.notify_one();
	if(isDeleted) deletedConsumer(std::vector<Image>());
}

void
ClassifierQueue::Impl::run()
{
	classifier.updateThreadSpecificSettings();
	while(isRunning) {
		std::pair<Image, Consumer> pair;
		bool isEmpty = true;
		{
			std::lock_guard<std::mutex> guard(mutex);
			if (!queue.empty()) {
				isEmpty = false;
				pair = queue.front();
				queue.pop_front();
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
