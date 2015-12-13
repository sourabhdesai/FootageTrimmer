#ifndef SVDESAI2_FINAL_FOOTAGETRIMMER_H
#define SVDESAI2_FINAL_FOOTAGETRIMMER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include "ImgSimilarityProcessor.h"

#define CLUSTER_SIZE 3
#define NUM_REFINEMENT_ITERATIONS 50

using namespace cv;
using std::vector;
using std::cout;
using std::endl;

class FootageTrimmer {
public:
    FootageTrimmer();
    FootageTrimmer(VideoCapture & trainingVideo);
	FootageTrimmer(Mat_<Vec3d> trainedPic);

    void saveToFile(string filepath);

    void displayTrainedImage(string windowName);

    vector<double> getTimeFrames() const;

	Size getFrameSize() const;

	class TrimmedFootage {
	public:
		void operator>>(Mat_<Vec3d> & outFrame);
		void operator>>(VideoWriter & videoWriter);

		long getNumSkippedFrames() const;

		TrimmedFootage(Mat_<Vec3d> avgFrame, VideoCapture & videoCapture, double similarityLowerBound);

	private:
		Mat_<Vec3d> avgFrame;
		VideoCapture videoCapture;
		const double similarityLowerBound;
		long numSkippedFrames;
	};

	FootageTrimmer::TrimmedFootage trim(VideoCapture & videoCapture, double tolerance);

private:
    Mat_<Vec3d> avgPicBuffer;
    Mat_<Vec3b> trainedPic;
    vector<double> frameTimes;

	Mat_<Vec3d> getBackgroundPic(VideoCapture & trainingVideo);

	/**
	 * Calculates values that indicates the amount of similarity between the two images.
	 */
	static double imageSimilarityVal(const Mat_<Vec3d> &imgA, const Mat_<Vec3d> &imgB);

	Mat_<Vec3d> getAveragePicture(const vector<Mat_<Vec3d> > &images);

    Mat_<Vec3d> findCloserImage(VideoCapture &trainingVideo, const vector<Mat_<Vec3d> > &images, double maxSimilarity, double *newFrameTime);

    /**
     * Removes image that is the most different from the other images
     * Image difference is found using imageDiffVal
     */
    int findMostDifferent(const vector<Mat_<Vec3d> > &images, double *img_difference);
};

#endif