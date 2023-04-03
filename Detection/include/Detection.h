#pragma once
#include<onnxruntime_cxx_api.h>
#include<opencv2/opencv.hpp>
#include<fstream>
#include<string>
#define confidenceRate 0.5
#define HEIGHT 640
#define WIDTH 640
struct detectedObject {
	cv::Rect box;
	float conf{};
	int classId{};
};
class YoloObject {
private:
	const wchar_t* model_path;
	float* blob = nullptr;
	std::string classes_path;
	std::vector<float> inputTensorValues;
	size_t inputTensorSize;
	std::vector<int64_t> inputTensorShape;
	Ort::Session session{ nullptr };
	Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	Ort::AllocatorWithDefaultOptions allocator;

	
	
	void getClassNames(std::vector<std::string>& class_names, std::string classFilePath);
	
	void vectorProductOfInputTensor();

	void preprocessing(const cv::Mat& frame);

	std::vector<detectedObject> decriptOutput(std::vector<Ort::Value>& output, const cv::Mat& frame);

	void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
		float& bestConf, int& bestClassId);
	
public:

	std::vector<std::string> classNames;
	YoloObject(const wchar_t* model_path, std::string class_path);
	std::vector<detectedObject> detect(const cv::Mat& frame);
};