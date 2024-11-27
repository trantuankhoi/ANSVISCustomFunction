#ifndef ANSCUSTOMCODE_H
#define ANSCUSTOMCODE_H
#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#define CUSTOM_API __declspec(dllexport)
#include "src/lite.h"

class CUSTOM_API IANSCustomClass
{
protected:
    std::string _modelDirectory; // The directory where the model is located
public:
    virtual bool Initialize(const std::string& modelDirectory, std::string& labelMap) = 0;
    virtual bool OptimizeModel(bool fp16) = 0;
    virtual std::vector<lite::types::CustomObject> RunInference(const cv::Mat& input) = 0;
    virtual bool Destroy() = 0;
};

#endif