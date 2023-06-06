#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/video/tracking.hpp>
#include<Detection.h>

using namespace std;

class Tracking {
private:
	string pathToBackbone= "Models/ModelsForTracking/nanotrack_backbone_sim.onnx";
	string pathToNeckhead= "Models/ModelsForTracking/nanotrack_head_sim.onnx";
	cv::TrackerNano::Params trackerParams;
	
	vector<cv::Ptr<cv::Tracker>> tracker;
	
	
public:
	Tracking();
	cv::Ptr<cv::Tracker> initObject(cv::Mat& frame, detectedObject& object);

	void initObjectsForTracking(cv::Mat& frame, vector<detectedObject>& objects);

	void updateTracker(cv::Mat& frame, vector<detectedObject>& objects);
};
