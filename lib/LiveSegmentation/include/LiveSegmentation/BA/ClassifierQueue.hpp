#ifndef LIVESEGMENTATION_CLASSIFIERQUEUE_H
#define LIVESEGMENTATION_CLASSIFIERQUEUE_H

#include <memory>

#include <opencv2/core/core.hpp>
#include "Classifier.hpp"

namespace ls {

typedef cv::Mat Image;
typedef std::function<void(std::vector<Image>)> Consumer;

class ClassifierQueue
{
	public:

		ClassifierQueue(Classifier& classifier);
		~ClassifierQueue();

		ClassifierQueue(ClassifierQueue && op) noexcept;
		ClassifierQueue& operator=(ClassifierQueue && op) noexcept;

		//ClassifierQueue(const ClassifierQueue& op);
		//ClassifierQueue& operator=(const ClassifierQueue& op);
		
		Classifier getClassifier();
		void start();
		void stop();
		void setLimit(size_t limit, bool overwrite=true);
		void add(Image image, Consumer consumer);

	private:
		class Impl;
		std::unique_ptr<Impl> impl;
};

}
#endif // LIVESEGMENTATION_CLASSIFIERQUEUE_H
