#include<Detection.h>
#include<thread>
using namespace std;

	void YoloObject::getClassNames(std::vector<std::string>& class_names, std::string classFilePath) {
		ifstream MyFile(classFilePath);
		string line;

		if (!MyFile.is_open()) {
			cout << "Couldn't open file with classes" << endl;
			return;
		}
		else {
			while (getline(MyFile, line)) {
				class_names.push_back(line);
			}
		}

	}


	YoloObject::YoloObject(const wchar_t* model_path,std::string class_path){
		this->model_path = model_path;
		this->classes_path = class_path;
		cout << "Get classes" << endl;
		getClassNames(this->classNames, this->classes_path);
		cout << "Make session" << endl;
		try {
			sessionOptions.SetIntraOpNumThreads(thread::hardware_concurrency()/2);
			
			cout << thread::hardware_concurrency() << endl;
			this->session = Ort::Session(env, model_path, sessionOptions);
		}
		catch (...) {
			cout << "Couldnt make session" << endl;
		}
	
		inputTensorShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
		vectorProductOfInputTensor();

	}

	

	void YoloObject::vectorProductOfInputTensor() {
		size_t product = 1;
		for (auto x : inputTensorShape) {
			product *= x;
		}
		inputTensorSize = product;

		cout << "Vector product of input tensor: " << inputTensorSize << endl;
	}
	void YoloObject::preprocessing(const cv::Mat& frame) {
		cv::Mat resizedImage, floatImage;
		
		cv::resize(frame, resizedImage, cv::Size(640, 640));
		cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);
		
		resizedImage.convertTo(floatImage, CV_32FC3, 1. / 255.0);
		
		blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];


			cv::Size floatImageSize{floatImage.cols, floatImage.rows};

			// hwc -> chw
			vector<cv::Mat> chw(floatImage.channels());

			for (int i = 0; i < floatImage.channels(); ++i)
			{
				chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
			}
			cv::split(floatImage, chw);
			
	}

	std::vector<detectedObject> YoloObject::detect(const cv::Mat& frame) {

		std::vector<const char*> inputNames, outputNames;
		
		auto input_name = session.GetInputNameAllocated(0, allocator);
		auto output_name = session.GetOutputNameAllocated(0, allocator);
		inputNames.push_back(input_name.get());
		outputNames.push_back(output_name.get());

		
		preprocessing(frame);

		auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		 inputTensorValues=std::vector<float>(blob, blob + inputTensorSize);

		std::vector<Ort::Value> inputTensors;

		inputTensors.push_back(Ort::Value::CreateTensor<float>(
			memoryInfo, inputTensorValues.data(), inputTensorSize,
			inputTensorShape.data(), inputTensorShape.size()
			));
		std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{ nullptr },
			inputNames.data(),
			inputTensors.data(),
			1,
			outputNames.data(),
			1);
		
		delete this->blob;
		return decriptOutput(outputTensors, frame);
	}

	std::vector<detectedObject> YoloObject::decriptOutput(std::vector<Ort::Value>& outputTensors,const cv::Mat& frame) {
		std::vector<detectedObject> data;
		std::vector<cv::Rect> boxes;
		std::vector<float> confs;
		std::vector<int> classIds;
		
		std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

		size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
		std::vector<float> output(outputTensors[0].GetTensorData<float>(), outputTensors[0].GetTensorData<float>() + count);

		int numClasses = (int)outputShape[1] - 4;
		int elementsInBatch = (int)(outputShape[1] * outputShape[2]);
		float r[2] = { (float)frame.size().width / 640,(float)frame.size().height / 640 };
		for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[1])
		{
			float clsConf = it[4];
			float objConf;
			int classId;
			//getBestClassInfo(it, numClasses, objConf, classId);
			
			if (clsConf > confidenceRate)
			{
				int centerX = (int)(it[0])*r[0];
				int centerY = (int)(it[1])*r[1];
				int width = (int)(it[2]) * r[0];
				int height = (int)(it[3]) * r[1];
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				float objConf = it[4];
				int classId = 0;


				for (int i = 5; i < numClasses + 5; i++)
				{
					if (it[i] > objConf)
					{
						objConf = it[i];
						classId = i - 5;
					}
				}
				
				float confidence = clsConf * objConf;

				boxes.emplace_back(left, top, width, height);
				confs.emplace_back(confidence);
				classIds.emplace_back(classId);


			}

			
		}

		vector<int> indices;
		cv::dnn::NMSBoxes(boxes, confs, confidenceRate, 0.5, indices);

		
		
		for (int idx : indices)
		{
			detectedObject det;
			det.box = cv::Rect(boxes[idx]);
			float* ratio = new float[2];
			
			

			
			det.conf = confs[idx];
			det.classId = classIds[idx];
			data.emplace_back(det);
		}


		for (const detectedObject& detection : data)
		{
			cv::rectangle(frame, detection.box, cv::Scalar(229, 160, 21), 2);

			int x = detection.box.x;
			int y = detection.box.y;
			
			int conf = (int)round(detection.conf * 100);
			int classId = detection.classId;


			cv::putText(frame, classNames[classId] + " conf: " + to_string(conf), cv::Point(x - 10, y - 10), 0.5, 0.5, cv::Scalar(200, 150, 20), 1);


		}

		return data;
	}


	void YoloObject::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
		float& bestConf, int& bestClassId)
	{
		// first 5 element are box and obj confidence
		bestClassId = 5;
		bestConf = 0;

		for (int i = 5; i < numClasses + 5; i++)
		{
			if (it[i] > bestConf)
			{
				bestConf = it[i];
				bestClassId = i - 5;
			}
		}

	}

	
