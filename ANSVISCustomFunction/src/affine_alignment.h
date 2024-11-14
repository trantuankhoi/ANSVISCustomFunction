// affine_alignment.hpp
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class AffineAlignment {
public:
    AffineAlignment(cv::Size image_size = cv::Size(112, 112));

    cv::Mat crop_image_by_mat(const cv::Mat& image, const std::vector<float>& landmarks);
    cv::Mat crop_image_by_bbox(const cv::Mat& image, const cv::Mat& bbox, float margin_ratio = 0.2);

private:
    cv::Size image_size_;
};