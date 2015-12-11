//
// Created by Sourabh Desai on 12/3/15.
//

#ifndef SVDESAI2_FINAL_IMGDIFFPROCESSOR_H
#define SVDESAI2_FINAL_IMGDIFFPROCESSOR_H

#include <opencv2/opencv.hpp>

using namespace cv;

class ImgDiffProcessor: public cv::ParallelLoopBody {

public:
    ImgDiffProcessor(const Mat_<Vec3d> &imgA, const Mat_<Vec3d> &imgB);
    ~ImgDiffProcessor();

    virtual void operator()( const cv::Range &r ) const;

    double getValue();

private:
    const Mat_<Vec3d> & imgA;
    const Mat_<Vec3d> & imgB;
    double *diffValues;
    double diff;
};


#endif //SVDESAI2_FINAL_IMGDIFFPROCESSOR_H
