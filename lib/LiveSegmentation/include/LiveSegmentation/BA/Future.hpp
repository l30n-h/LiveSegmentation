#ifndef LIVESEGMENTATION_FUTURE_H
#define LIVESEGMENTATION_FUTURE_H

#include <memory>

namespace ls {

template<typename T> 
class Future
{
	public:
        Future();
		~Future();

		bool isDone();
		bool isValid();
		T get();
        void set(const T&);
		void cancel();

	private:
		class Impl;
		std::shared_ptr<Impl> impl;
};

}

#include "impl/FutureDetail.hpp"

#endif // LIVESEGMENTATION_FUTURE_H
