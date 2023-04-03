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
    auto centroidtracker = new CentroidTracker(20);
    for (int i = 0; i < 1; i++)
        video >> frame;
    std::chrono::milliseconds duration;
    while (1) {
        video >> frame;
        second = frame.clone();
        auto start = std::chrono::high_resolution_clock::now();
        vector<detectedObject> objekti;
        vector<vector<int>> boxes;
        objekti=detection.detect(frame);
        for (auto x : objekti) {
            boxes.insert(boxes.end(), { x.box.x,x.box.y,x.box.width,x.box.height });
        }

        auto objects = centroidtracker->update(boxes);

        auto stop = std::chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cv::putText(frame, to_string(1000/duration.count()), cv::Point(30, 40),1, 1.8, cv::Scalar(200, 200, 200),3);

        for (auto obj : objects) {
            int k = 1;
            for (int i = 1; i < centroidtracker->path_keeper[obj.first].size(); i++) {
                int thickness = int(sqrt(20 / float(k + 1) * 2.5));
                cv::line(frame,
                    cv::Point(centroidtracker->path_keeper[obj.first][i - 1].first, centroidtracker->path_keeper[obj.first][i - 1].second),
                    cv::Point(centroidtracker->path_keeper[obj.first][i].first, centroidtracker->path_keeper[obj.first][i].second),
                    cv:: Scalar(0, 0, 255), thickness);
                k += 1;
            }
        }
    
    cv::imshow("asd", frame);
        if (cv::waitKey(1) == 'q')
            return 0;
        
       
       
    }
}

