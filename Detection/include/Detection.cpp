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

	void YoloObject::detect(const cv::Mat& frame,std::vector<detectedObject>& objects) {

		std::vector<const char*> inputNames, outputNames;
		objects.clear();
		
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
		decriptOutput(outputTensors, frame,objects);
	}

	void YoloObject::decriptOutput(std::vector<Ort::Value>& outputTensors,const cv::Mat& frame, std::vector<detectedObject>& data) {
		
		std::vector<cv::Rect> boxes;
		std::vector<float> confs;
		std::vector<int> classIds;
		
		std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

		size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
		std::vector<float> output(outputTensors[0].GetTensorData<float>(), outputTensors[0].GetTensorData<float>() + count);

		cv::Mat l_Mat = cv::Mat(outputShape[1], outputShape[2], CV_32FC1, (void*)outputTensors[0].GetTensorData<float>());
		cv::Mat l_Mat_t = l_Mat.t();

		int numClasses = l_Mat_t.cols - 4;
		int elementsInBatch = (int)(outputShape[1] * outputShape[2]);
		float r[2] = { (float)frame.size().width / 640,(float)frame.size().height / 640 };

		for (int l_Row = 0; l_Row < l_Mat_t.rows; l_Row++)
		{
			cv::Mat l_MatRow = l_Mat_t.row(l_Row);
			float objConf;
			int classId;

			getBestClassInfo(l_MatRow, numClasses, objConf, classId);
			
			if (objConf > confidenceRate)
			{
				float centerX = (l_MatRow.at<float>(0, 0));
				float centerY = (l_MatRow.at<float>(0, 1));
				float width = (l_MatRow.at<float>(0, 2));
				float height = (l_MatRow.at<float>(0, 3));
				float left = centerX - width / 2;
				float top = centerY - height / 2;

				


				float confidence = objConf;
				
				

				boxes.emplace_back(left, top, width, height);
				confs.emplace_back(confidence);
				classIds.emplace_back(classId);


			}

			
		}

		vector<int> indices;
		cv::dnn::NMSBoxes(boxes, confs, confidenceRate, 0.3, indices);

		
		
		for (int idx : indices)
		{
			detectedObject det;
			det.box = cv::Rect(boxes[idx]);
			float* ratio = new float[2];
			det.conf = confs[idx];
			det.classId = classIds[idx];
			data.emplace_back(det);
		}


		
	}


	void YoloObject::getBestClassInfo(const cv::Mat& p_Mat, const int& numClasses,
		float& bestConf, int& bestClassId)
	{
		bestClassId = 0;
		bestConf = 0;

		if (p_Mat.rows && p_Mat.cols)
		{
			for (int i = 0; i < numClasses; i++)
			{
				if (p_Mat.at<float>(0, i + 4) > bestConf)
				{
					bestConf = p_Mat.at<float>(0, i + 4);
					bestClassId = i;
				}
			}
		}
	}

	void YoloObject::drawDetectedObjectaOnFrame(cv::Mat& frame, const std::vector<detectedObject>& objects) {

		for (const detectedObject& detection : objects)
		{
			cv::rectangle(frame, detection.box, cv::Scalar(229, 160, 21), 2);

			int x = detection.box.x;
			int y = detection.box.y;

			int conf = (int)round(detection.conf * 100);
			int classId = detection.classId;

			cv::putText(frame, classNames[classId] + " conf: " + to_string(conf), cv::Point(x - 10, y - 10), 0.5, 0.5, cv::Scalar(200, 150, 20), 1);


		}


	}
