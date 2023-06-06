#include <iostream>
#include<Detection.h>
#include<chrono>
#include<Tracking.h>
#define FRAMES_TO_SKIP 10

using namespace std;

const string FilePath = "Files/";
const string LiveStream = "http://192.168.0.193:4747/video";
int main()
{
    cv::Mat frame;
    int frameCount=0;

   


    //cv::VideoCapture video("E:/Luka_Split_Ljeto/datadir0/hiv00118.mp4");
    //cv::VideoCapture video("E:/ACI/vlc-record-2023-02-15-12h34m33s-rtsp___195.29.155.14-.mp4");
    cv::VideoCapture video("Files/video2.mp4");
    //cv::VideoCapture video(0);
    YoloObject* detection=new YoloObject(L"Models/bestv5.onnx", FilePath + "classes.txt");
    Tracking* tracker = new Tracking();

    //for (int i = 0; i < 100; i++)
      //  video >> frame;

    video >> frame;

    frameCount++;

    cv::resize(frame.clone(), frame, cv::Size(640, 640));
    vector<detectedObject> objekti;
    detection->detect(frame, objekti);
    tracker->initObjectsForTracking(frame, objekti);
    std::chrono::milliseconds duration;
    while (1) {

        
        objekti.clear();
        video >> frame;
        
        frameCount++;

        cv::resize(frame.clone(), frame, cv::Size(640, 640));
        
        auto start = std::chrono::high_resolution_clock::now();

        //tracker->updateTracker(frame, objekti);
        switch (cv::waitKey(1))
        {
        case 'q':
            //Izgasi aplikaciju
            delete detection;
            return 0;
        case 's':
            //Obavi detekciju
            objekti.clear();
           detection->detect(frame, objekti);
            detection->drawDetectedObjectaOnFrame(frame, objekti);

            break;
        case 'd':
            //Premotaj u naprid za 100 frame-ova
            frameCount = frameCount + FRAMES_TO_SKIP;
            video.set(cv::CAP_PROP_POS_FRAMES, frameCount);
            break;
        case 'a':
            //Premotaj u natrag za 100 frame-ova
            if (frameCount > 100) {
                frameCount = frameCount - FRAMES_TO_SKIP;
                video.set(cv::CAP_PROP_POS_FRAMES, frameCount - FRAMES_TO_SKIP);
            }
            break;
        default:
            
            detection->detect(frame, objekti);
            //detection->drawDetectedObjectaOnFrame(frame, objekti);
            break;

        }

        detection->drawDetectedObjectaOnFrame(frame, objekti);
        auto stop = std::chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        
        cv::putText(frame, to_string(duration.count()), cv::Point(30, 40), 1, 1.8, cv::Scalar(200, 200, 200), 3);
        
        cv::imshow("asd", frame);
       
        
        
        

            
            
        
       
       
    }
}

