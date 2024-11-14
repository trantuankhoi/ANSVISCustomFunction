// affine_alignment.cpp
#include "affine_alignment.h"

AffineAlignment::AffineAlignment(cv::Size image_size)
    : image_size_(image_size) {
}

cv::Mat AffineAlignment::crop_image_by_mat(const cv::Mat& image, const std::vector<float>& landmarks) {
    // Reshape landmarks to standard shape
    cv::Mat landmarks_mat(landmarks.size() / 2, 2, CV_32F);
    for (size_t i = 0; i < landmarks.size() / 2; i++) {
        landmarks_mat.at<float>(i, 0) = landmarks[i * 2];
        landmarks_mat.at<float>(i, 1) = landmarks[i * 2 + 1];
    }

    if (landmarks_mat.rows != 68 && landmarks_mat.rows != 5) {
        throw std::runtime_error("Number of landmarks must be 68 or 5");
    }

    cv::Mat landmark5(5, 2, CV_32F);
    if (landmarks_mat.rows == 68) {
        // Convert 68 landmarks to 5 landmarks
        landmark5.at<float>(0, 0) = (landmarks_mat.at<float>(36, 0) + landmarks_mat.at<float>(39, 0)) / 2;
        landmark5.at<float>(0, 1) = (landmarks_mat.at<float>(36, 1) + landmarks_mat.at<float>(39, 1)) / 2;
        landmark5.at<float>(1, 0) = (landmarks_mat.at<float>(42, 0) + landmarks_mat.at<float>(45, 0)) / 2;
        landmark5.at<float>(1, 1) = (landmarks_mat.at<float>(42, 1) + landmarks_mat.at<float>(45, 1)) / 2;
        landmark5.at<float>(2, 0) = landmarks_mat.at<float>(30, 0);
        landmark5.at<float>(2, 1) = landmarks_mat.at<float>(30, 1);
        landmark5.at<float>(3, 0) = landmarks_mat.at<float>(48, 0);
        landmark5.at<float>(3, 1) = landmarks_mat.at<float>(48, 1);
        landmark5.at<float>(4, 0) = landmarks_mat.at<float>(54, 0);
        landmark5.at<float>(4, 1) = landmarks_mat.at<float>(54, 1);
    }
    else {
        landmark5 = landmarks_mat.clone();
    }

    // Define source points
    cv::Mat src = (cv::Mat_<float>(5, 2) <<
        30.2946f, 51.6963f,
        65.5318f, 51.5014f,
        48.0252f, 71.7366f,
        33.5493f, 92.3655f,
        62.7299f, 92.2041f);

    // Scale source points
    src *= image_size_.width / 112.0f;
    src.col(0) += 8.0f;

    // Calculate transformation matrix
    cv::Mat M = cv::estimateAffinePartial2D(landmark5, src);

    // Apply affine transformation
    cv::Mat output;
    cv::warpAffine(image, output, M, image_size_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    return output;
}

cv::Mat AffineAlignment::crop_image_by_bbox(const cv::Mat& image, const cv::Mat& bbox, float margin_ratio) {
    float x1 = bbox.at<float>(0, 0);
    float y1 = bbox.at<float>(0, 1);
    float x2 = bbox.at<float>(0, 2);
    float y2 = bbox.at<float>(0, 3);

    int width = static_cast<int>(x2 - x1);
    int height = static_cast<int>(y2 - y1);
    int maximum = std::max(width, height);
    int margin = static_cast<int>(maximum * margin_ratio);
    int dx = static_cast<int>((maximum - width) / 2);
    int dy = static_cast<int>((maximum - height) / 2);

    // Adjust coordinates with margin and bounds checking
    x1 = std::max(0.0f, x1 - dx - margin);
    y1 = std::max(0.0f, y1 - dy - margin);
    x2 = std::min(static_cast<float>(image.cols), x2 + dx + margin);
    y2 = std::min(static_cast<float>(image.rows), y2 + dy + margin);

    // Crop the image
    cv::Mat face = image(cv::Range(static_cast<int>(y1), static_cast<int>(y2)),
        cv::Range(static_cast<int>(x1), static_cast<int>(x2)));
    return face;
}