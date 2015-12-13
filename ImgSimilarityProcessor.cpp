//
// Created by Sourabh Desai on 12/3/15.
//

#include "ImgSimilarityProcessor.h"

ImgSimilarityProcessor::ImgSimilarityProcessor(const Mat_<Vec3d> &imgA, const Mat_<Vec3d> &imgB) : imgA(imgA), imgB(imgB) {
    this->diffValues = new double[imgA.rows * imgA.cols];
    memset(this->diffValues, 0x0, imgA.rows * imgA.cols * sizeof(double));
    this->diff = -1;
}

void ImgSimilarityProcessor::operator()(const cv::Range &r) const {
    int row, col;
    for (row = r.start; row < r.end; row++) {
        for (col = 0; col < this->imgA.cols; col++) {
            const Vec3d & pixelA = this->imgA.at<Vec3d>(row, col);
            const Vec3d & pixelB = this->imgB.at<Vec3d>(row, col);

            double similarity = pixelA.dot(pixelB) / (norm(pixelA) * norm(pixelB));

            size_t diffValuesIdx = (size_t) ((row * imgA.cols) + col);
            this->diffValues[diffValuesIdx] = similarity < 0.0 ? 0.0 : similarity;
        }
    }
}

double ImgSimilarityProcessor::getValue() {
    if (this->diff >= 0) {
        return this->diff;
    }

    this->diff = 0;
    size_t idx;

    for (idx=0; idx < (this->imgA.rows * this->imgA.cols); idx++) {
        this->diff += isnan(this->diffValues[idx]) ? 1.0 : this->diffValues[idx];
//        if (isnan(this->diffValues[idx])) {
//            std::cout << "isnan==true:\t" << this->diffValues[idx] << "\nidx:\t" << idx << std::endl;
//        }
    }

    return this->diff;
}

ImgSimilarityProcessor::~ImgSimilarityProcessor() {
    delete [] this->diffValues;
}
