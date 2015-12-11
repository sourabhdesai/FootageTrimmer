//
// Created by Sourabh Desai on 12/3/15.
//

#include "ImgDiffProcessor.h"

ImgDiffProcessor::ImgDiffProcessor(const Mat_<Vec3d> &imgA, const Mat_<Vec3d> &imgB) : imgA(imgA), imgB(imgB) {
    this->diffValues = new double[imgA.rows * imgA.cols];
    this->diff = -1;
}

void ImgDiffProcessor::operator()(const cv::Range &r) const {
    int idx;
    for (idx = r.start; idx < r.end; idx++) {
        const Vec3d & pixelA = this->imgA.at<Vec3d>(idx);
        const Vec3d & pixelB = this->imgB.at<Vec3d>(idx);

        double similarity = pixelA.dot(pixelB) / (norm(pixelA) * norm(pixelB));

        this->diffValues[idx] = similarity < 0.0 ? 0.0 : similarity;
    }
}

double ImgDiffProcessor::getValue() {
    if (this->diff >= 0) {
        return this->diff;
    }

    this->diff = 0;
    size_t idx;

    for (idx=0; idx < (this->imgA.rows * this->imgA.cols); idx++) {
        this->diff += this->diffValues[idx];
    }

    return this->diff;
}

ImgDiffProcessor::~ImgDiffProcessor() {
    delete [] this->diffValues;
}
