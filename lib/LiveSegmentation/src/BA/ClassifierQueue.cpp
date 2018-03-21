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
		ClassifierFuture add(const Image& image);
		void run();

		size_t limit = 1;
		bool overwrite = true;
		bool isRunning = false;
		std::mutex mutex;
		std::mutex conditionMutex;
		std::condition_variable conditionVariable;
		std::unique_ptr<std::thread> workerThread;

		std::deque< std::pair<Image, ClassifierFuture> > queue;
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

ClassifierFuture
ClassifierQueue::add(const Image& image)
{
	return impl->add(image);
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
			std::pair<Image, ClassifierFuture> pair;
			while(queue.size()>pLimit){
				pair = queue.back();
				queue.pop_back();
				pair.second.cancel();
			}
		}
		limit = pLimit;
		overwrite=pOverwrite;
	}
}

ClassifierFuture
ClassifierQueue::Impl::add(const Image& image)
{
	ClassifierFuture deletedFuture;
	bool isDeleted = false;
	bool isFull = false;
	auto future = ClassifierFuture();
	{
		std::lock_guard<std::mutex> guard(mutex);
		isFull=queue.size()==limit;
		if(isFull) {
			if(overwrite){
				deletedFuture = queue.back().second;
				queue.pop_back();
				isFull = false;
			}else {
				deletedFuture = future;
			}
			isDeleted = true;
		}
		if(!isFull) queue.push_back(std::make_pair(image, future));
	}
	if(!isFull) conditionVariable.notify_one();
	if(isDeleted){
		deletedFuture.cancel();
	}
	return future;
}

void
ClassifierQueue::Impl::run()
{
	classifier.updateThreadSpecificSettings();
	while(isRunning) {
		std::pair<Image, ClassifierFuture> pair;
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
			pair.second.set(classifier.Predict(pair.first));
		}
	}
}

}
