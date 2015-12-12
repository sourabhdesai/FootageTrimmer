#include <iostream>
#include <opencv2/opencv.hpp>

#include "FootageTrimmer.h"

using std::cout;
using std::endl;
using std::string;

using namespace cv;

#define TOLERANCE_FACTOR 0.66

/**
 * Command Line Options:
 * -v: path to video to time compress
 * -o: path to save the time compressed video
 * -t: path to trained picture to use in compression
 * -s: path to save the training picture to
 * -p: verbose option; will print out time frames from algorithm
 */
int main(int argc, char *argv[]) {
    if (argc <= 1) {
        cout << "Program requires arg to specify path to video" << endl;
        return EXIT_FAILURE;
    }

    char cwdBuff[1000];

    string cwd = string(getcwd(cwdBuff, 1000));
    string videoPath;
    string compressedVideoPath;
    string trainedPicPath;
    string trainedPicSavePath;
    bool verbose = false;

    // Parse arguments
    char c;
    while ((c = getopt(argc, argv, "v:o:t:s:p")) != -1) {
        switch (c) {
            case 'v': {
                videoPath = string(optarg);
                break;
            }
            case 'o': {
                compressedVideoPath = string(optarg);
                break;
            }
            case 't': {
                trainedPicPath = string(optarg);
                break;
            }
            case 's': {
                trainedPicSavePath = string(optarg);
                break;
            }
            case 'p': {
                verbose = true;
                break;
            }
            default: {
                cout << "Invalid option: " << char(optopt) << endl;
                return EXIT_FAILURE;
            }
        }
    }

    if (trainedPicPath.length() && trainedPicSavePath.length()) {
        cout << "Can't have -t and -s options together" << endl;
        return EXIT_FAILURE;
    }

    if (videoPath.length() && compressedVideoPath.empty() && trainedPicSavePath.empty()) {
        cout << "If -v specified, must specify either -o or -s" << endl;
        return EXIT_FAILURE;
    }

    VideoCapture inputVideo(videoPath);

    FootageTrimmer footageTrimmer;

    if (trainedPicPath.length()) {
        Mat_<Vec3b> trainedPic = imread(trainedPicPath);
        Mat_<Vec3d> trainedPicDouble;
        trainedPic.convertTo(trainedPicDouble, CV_64F);
        footageTrimmer = FootageTrimmer(trainedPicDouble); // train the footage trimmer from a saved instance
    } else {
        footageTrimmer = FootageTrimmer(inputVideo); // train the footage trimmer from the given video
    }

    inputVideo.set(CV_CAP_PROP_POS_AVI_RATIO, 0); // rewind the footage to the start


    if (trainedPicSavePath.length()) {
        footageTrimmer.saveToFile(trainedPicSavePath);
    }

    if (verbose) {
        vector<double> timeFrames = footageTrimmer.getTimeFrames();
        cout << "Time Frames: [\t";
        vector<double>::const_iterator it;
        for (it = timeFrames.begin(); it != timeFrames.end(); it++) {
            cout << *it << "\t";
        }
        cout << "]" << endl;
    }

    if (compressedVideoPath.length()) {
        double fps = inputVideo.get(CV_CAP_PROP_FPS);
        Size frameSize = footageTrimmer.getFrameSize();
        VideoWriter outputVideo(compressedVideoPath, CV_FOURCC('m', 'p', '4', 'v'), fps, frameSize);
        FootageTrimmer::TrimmedFootage trimmedFootage = footageTrimmer.trim(inputVideo, TOLERANCE_FACTOR * frameSize.area());
        trimmedFootage >> outputVideo;
    }


    return EXIT_SUCCESS;
}
/*
int main(int argc, char *argv[]) {

    VideoCapture cap = VideoCapture(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat edges;
    Mat edges2;
    int c = 0;
    namedWindow("edges",1);
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        if (c++ == 0) cout << "frame.type:\t" << frame.type() << endl;
        cvtColor(frame, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        // Canny(edges, edges, 0, 30, 3);
        if (edges2.total()) {
            absdiff(edges, edges2, edges);
            imshow("edges", edges);
        }
        
        edges.copyTo(edges2);

        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}*/
