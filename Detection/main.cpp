#include <iostream>
#include<Detection.h>
#include<chrono>
#include "centroidtracker.h"

using namespace std;

const string FilePath = "Files/";
const string LiveStream = "http://192.168.0.193:4747/video";
int main()
{
    cv::Mat frame,second;
    cv::VideoCapture video(0);
    YoloObject detection(L"Models/yolov5n.onnx", FilePath + "classes.txt");
    
    for (int i = 0; i < 100; i++)
        video >> frame;
    
    std::chrono::milliseconds duration;
    while (1) {
        video >> frame;
        cv::resize(frame.clone(), frame, cv::Size(640, 640));
        second = frame.clone();
        auto start = std::chrono::high_resolution_clock::now();
        vector<detectedObject> objekti;
        vector<vector<int>> boxes;
        objekti=detection.detect(frame);
        for (auto x : objekti) {
            boxes.insert(boxes.end(), { x.box.x,x.box.y,x.box.width,x.box.height });
        }

        

        auto stop = std::chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cv::putText(frame, to_string(1000/duration.count()), cv::Point(30, 40),1, 1.8, cv::Scalar(200, 200, 200),3);

       
    cv::imshow("asd", frame);
        if (cv::waitKey(1) == 'q')
            return 0;
        
       
       
    }
}

