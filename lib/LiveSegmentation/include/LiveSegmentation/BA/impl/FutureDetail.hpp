#ifndef LIVESEGMENTATION_FUTUREDETAIL_H
#define LIVESEGMENTATION_FUTUREDETAIL_H

#include "LiveSegmentation/BA/Future.hpp"

#include <vector>
#include <mutex>
#include <condition_variable>

namespace ls {

template<typename T> 
class Future<T>::Impl
{
	public:
        Impl();
        ~Impl();

		T get();
		void set(const T&);
		void cancel();

		bool isDone = false;
		bool isValid = false;
		T result;

		std::mutex conditionMutex;
    	std::condition_variable conditionVariable;
};

template<typename T>
Future<T>::Future()
:impl(std::make_shared<Impl>())
{

}

template<typename T>
Future<T>::~Future()
{

}

template<typename T>
bool
Future<T>::isDone()
{
	return impl->isDone;
}

template<typename T>
bool
Future<T>::isValid()
{
	return impl->isValid;
}

template<typename T>
T
Future<T>::get()
{
	return impl->get();
}

template<typename T>
void
Future<T>::set(const T& r){
    impl->set(r);
}

template<typename T>
void
Future<T>::cancel()
{
	impl->cancel();
}

template<typename T>
Future<T>::Impl::Impl()
{
	
}

template<typename T>
Future<T>::Impl::~Impl()
{
	cancel();
}

template<typename T>
T
Future<T>::Impl::get(){
	if(!isDone){
		std::unique_lock<std::mutex> conditionLock(conditionMutex);
		conditionVariable.wait(conditionLock);
	}
	return result;
}

template<typename T>
void
Future<T>::Impl::cancel(){
    if(!isDone){
        //result = T();
        isValid = false;
        isDone = true;
        conditionVariable.notify_all();
    }
}

template<typename T>
void
Future<T>::Impl::set(const T& r){
    if(!isDone){
        result = r;
        isValid = true;
        isDone = true;
        conditionVariable.notify_all();
    }
}

}
#endif // LIVESEGMENTATION_FUTUREDETAIL_H
