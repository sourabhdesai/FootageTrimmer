#include "FootageTrimmer.h"

FootageTrimmer::FootageTrimmer(VideoCapture & trainingVideo) {
	Mat_<Vec3d> backPicDouble = this->getBackgroundPic(trainingVideo);
    backPicDouble.convertTo(this->trainedPic, CV_8UC3);
}

Mat_<Vec3d> FootageTrimmer::getBackgroundPic(VideoCapture & trainingVideo) {
    if (!this->trainedPic.empty()) {
        return this->trainedPic;
    }

	vector<Mat_<Vec3d> > cluster;
	cluster.reserve(CLUSTER_SIZE);

	// Initially set all cluster images so they are evenly spaced out throughout the video
	double clusterTimeIdx; // time interpolant that goes from 0 (start of video) to 1 (end of video)
    for (clusterTimeIdx=0; clusterTimeIdx <= 1.0 && cluster.size() < CLUSTER_SIZE; clusterTimeIdx += 1.0/double(CLUSTER_SIZE)) {
        trainingVideo.set(CV_CAP_PROP_POS_AVI_RATIO, clusterTimeIdx);
        Mat_<Vec3b> intImg;
        Mat_<Vec3d> doubleImg;
        trainingVideo >> intImg;
        intImg.convertTo(doubleImg, CV_64F);
//		cvtColor(img, img, CV_BGR2RGB);
		cluster.push_back(doubleImg);
        this->frameTimes.push_back(clusterTimeIdx);
	}

    // Perform image set refinement
    int prune_count;
    for (prune_count = 0; prune_count < NUM_REFINEMENT_ITERATIONS; prune_count++) {
        double difference;
        int most_diff_img_idx = this->findMostDifferent(cluster, &difference);
        cluster.erase(cluster.begin() + most_diff_img_idx);
        this->frameTimes.erase(this->frameTimes.begin() + most_diff_img_idx);
        double newFrameTime;
        Mat_<Vec3d> closerImg = this->findCloserImage(trainingVideo, cluster, difference, &newFrameTime);
        cluster.push_back(closerImg);
        this->frameTimes.push_back(newFrameTime);
        this->avgPicBuffer.release();
    }

    return this->getAveragePicture(cluster);
}

int FootageTrimmer::findMostDifferent(const vector<Mat_<Vec3d> > & images, double *img_difference) {
    this->avgPicBuffer = this->avgPicBuffer.empty() ? this->getAveragePicture(images) : this->avgPicBuffer;
    int mostDifferentImgIdx = 0;
    double maxDifference = this->imageSimilarityVal(images[0], this->avgPicBuffer);

    int img_idx;
    for (img_idx = 1; img_idx < images.size(); img_idx++) {
		Mat_<Vec3d> img = images[img_idx];
		double difference = this->imageSimilarityVal(img, this->avgPicBuffer);

        if (maxDifference < difference) {
            maxDifference = difference;
            mostDifferentImgIdx = img_idx;
        }
	}

    *img_difference = maxDifference;
	return mostDifferentImgIdx;
}

Mat_<Vec3d> FootageTrimmer::findCloserImage(VideoCapture & trainingVideo, const vector<Mat_<Vec3d> > & images, double maxSimilarity, double *newFrameTime) {
    this->avgPicBuffer = this->avgPicBuffer.empty() ? this->getAveragePicture(images) : this->avgPicBuffer;
    Mat_<Vec3b> candidateIntImg;
    Mat_<Vec3d> candidateImg;

    do {
        double randTime = double(rand()) / double(RAND_MAX);
        *newFrameTime = randTime;
        trainingVideo.set(CV_CAP_PROP_POS_AVI_RATIO, randTime);
        trainingVideo >> candidateIntImg;
        candidateIntImg.convertTo(candidateImg, CV_64F);
//        cvtColor(candidateImg, candidateImg, CV_BGR2RGB);
    } while(this->imageSimilarityVal(candidateImg, this->avgPicBuffer) > maxSimilarity);

    return candidateImg; // this image passed the similarity threshold
}

Mat_<Vec3d> FootageTrimmer::getAveragePicture(const vector<Mat_<Vec3d> > & images) {

    int img_h = images[0].rows;
    int img_w = images[0].cols;

    Mat_<Vec3d> averageImage(img_h, img_w);

    vector<Mat_<Vec3d> >::const_iterator it;

    for (it = images.begin(); it != images.end(); it++) {
        Mat_<Vec3d> img = *it;
        accumulate(img, averageImage);
    }

    double numImages = double(images.size());
    averageImage = averageImage / numImages;
    return averageImage;
}

double FootageTrimmer::imageSimilarityVal(const Mat_<Vec3d> &imgA, const Mat_<Vec3d> &imgB) {
    Mat_<Vec3d> imgABlurred;
    Mat_<Vec3d> imgBBlurred;

    GaussianBlur(imgA, imgABlurred, Size(7,7), 1.5, 1.5);
    GaussianBlur(imgB, imgBBlurred, Size(7,7), 1.5, 1.5);
    ImgSimilarityProcessor simProcessor(imgABlurred, imgBBlurred);
    int numPixels = imgA.rows * imgA.cols;
    parallel_for_(Range(0,numPixels), simProcessor);
    return simProcessor.getValue();
}

FootageTrimmer::FootageTrimmer(Mat_<Vec3d> trainedPic) {
    this->trainedPic = trainedPic;
}

FootageTrimmer::FootageTrimmer() {

}

void FootageTrimmer::saveToFile(string filepath) {
    if (this->trainedPic.empty()) {
        throw "This footagetrimmer instance hasn't been trained with any data";
    }

    imwrite(filepath, this->trainedPic);
}

void FootageTrimmer::displayTrainedImage(string windowName) {
    if (this->trainedPic.empty()) {
        throw "This footagetrimmer instance hasn't been trained with any data";
    }

    imshow(windowName, this->trainedPic);
}

vector<double> FootageTrimmer::getTimeFrames() const {
    return this->frameTimes;
}

Size FootageTrimmer::getFrameSize() const{
    return this->trainedPic.size();
}

FootageTrimmer::TrimmedFootage::TrimmedFootage(Mat_<Vec3d> avgFrame, VideoCapture & videoCapture, double similarityLowerBound):
        avgFrame(avgFrame), videoCapture(videoCapture), similarityLowerBound(similarityLowerBound), numSkippedFrames(0) {
    // nothing else to do other than initialization list
}

void FootageTrimmer::TrimmedFootage::operator>>(Mat_<Vec3d> &outFrame) {
    Mat_<Vec3b> intFrame;
    bool shouldSkipFrame;
    do {
        this->videoCapture >> intFrame;
        intFrame.convertTo(outFrame, CV_64F);
        if (!outFrame.empty()) {
            double similarity = FootageTrimmer::imageSimilarityVal(outFrame, this->avgFrame);
            shouldSkipFrame = similarity > this->similarityLowerBound;
        } else {
            shouldSkipFrame = false;
        }

        this->numSkippedFrames += shouldSkipFrame;
    } while (shouldSkipFrame);
}

void FootageTrimmer::TrimmedFootage::operator>>(VideoWriter &videoWriter) {
    Mat_<Vec3d> frame;
    Mat_<Vec3b> intFrame;
    (*this) >> frame;
    long frameCount = 0;
    while (!frame.empty()) {
        if ((frameCount++ % 25) == 0) {
            double videoPos = this->videoCapture.get(CV_CAP_PROP_POS_AVI_RATIO);
            cout << "Video Position: " << videoPos << endl;
            cout << "Curr Num Frames Skipped: " << this->numSkippedFrames << endl;
        }
        frame.convertTo(intFrame, CV_8UC3);
        videoWriter << intFrame;
        (*this) >> frame;
    }
}

FootageTrimmer::TrimmedFootage FootageTrimmer::trim(VideoCapture & videoCapture, double tolerance) {
    return FootageTrimmer::TrimmedFootage(this->trainedPic, videoCapture, tolerance);
}

long FootageTrimmer::TrimmedFootage::getNumSkippedFrames() const {
    return this->numSkippedFrames;
}
