#include "Tracking.h"


	Tracking::Tracking() {
		trackerParams.backbone = pathToBackbone;
		trackerParams.neckhead = pathToNeckhead;
	}
	cv::Ptr<cv::Tracker> Tracking::initObject(cv::Mat& frame, detectedObject& object) {
		cv::Ptr<cv::Tracker> track = cv::TrackerMIL::create();
		track->init(frame, object.box);
		return track;
	}

	void Tracking::initObjectsForTracking(cv::Mat& frame, vector<detectedObject>& objects) {
		for (int i = 0; i < objects.size(); i++) {
			tracker.push_back(initObject(frame, objects[i]));

		}
	}

	void Tracking::updateTracker(cv::Mat& frame, vector<detectedObject>& objects) {
		for (int i = 0; i < tracker.size(); i++) {
			tracker[i]->update(frame, objects[i].box);

		}
	}
