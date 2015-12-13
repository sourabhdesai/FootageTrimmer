//
// Created by Sourabh Desai on 12/3/15.
//

#ifndef SVDESAI2_FINAL_IMGSIMPROCESSOR_H
#define SVDESAI2_FINAL_IMGSIMPROCESSOR_H

#include <opencv2/opencv.hpp>

using namespace cv;

class ImgSimilarityProcessor : public cv::ParallelLoopBody {

public:
    ImgSimilarityProcessor(const Mat_<Vec3d> &imgA, const Mat_<Vec3d> &imgB);
    ~ImgSimilarityProcessor();

    virtual void operator()( const cv::Range &r ) const;

    double getValue();

private:
    const Mat_<Vec3d> & imgA;
    const Mat_<Vec3d> & imgB;
    double *diffValues;
    double diff;
};


#endif //SVDESAI2_FINAL_IMGSIMPROCESSOR_H
