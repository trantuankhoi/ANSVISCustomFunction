//
// Created by DefTruth on 2021/3/18.
//

#ifndef LITE_AI_ORT_ORT_CORE_H
#define LITE_AI_ORT_ORT_CORE_H

#include "ort_config.h"
#include "ort_handler.h"
#include "ort_types.h"

namespace ortcv
{
    class LITE_EXPORTS SCRFD;                      // [72] * reference: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
}

namespace ortnlp
{
    class LITE_EXPORTS TextCNN; // todo
    class LITE_EXPORTS ChineseBert; // todo
    class LITE_EXPORTS ChineseOCRLite; // todo
}

namespace ortsd
{
    class LITE_EXPORTS Clip; // [1] * reference: https://github.com/openai/CLIP
}

namespace ortcv
{
    using core::BasicOrtHandler;
    using core::BasicMultiOrtHandler;
}
namespace ortnlp
{
    using core::BasicOrtHandler;
    using core::BasicMultiOrtHandler;
}
namespace ortasr
{
    using core::BasicOrtHandler;
    using core::BasicMultiOrtHandler;
}

namespace ortsd
{
    using core::BasicOrtHandler;
    using core::BasicMultiOrtHandler;
}

#endif //LITE_AI_ORT_ORT_CORE_H
