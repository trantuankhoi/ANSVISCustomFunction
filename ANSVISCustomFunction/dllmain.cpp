#include "pch.h"
#include "ANSCustomCode.h"
#include "src/face_database.h"
#include "src/lite.h"

// The zip password to zip the customised model: AnsCustomModels20@$
class CUSTOM_API ANSCustomClass : public IANSCustomClass
{
public:
    bool Initialize(const std::string& modelDiretory, std::string& labelMap)override;
    bool OptimizeModel(bool fp16)override;
    std::vector<CustomObject> RunInference(const cv::Mat& input)override;
    bool Destroy()override;
    ANSCustomClass();
    ~ANSCustomClass();
};
BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

ANSCustomClass::ANSCustomClass()
{
    // Initialize the model
}
bool ANSCustomClass::OptimizeModel(bool fp16)
{
    // Optimize the model
    // User can access to the _modelDirectory to get the models' path
    // User can start doing the optimization here for each model
    return true;
}

std::vector<CustomObject> ANSCustomClass::RunInference(const cv::Mat& input)
{
    // Prepare output vector for detected faces
    std::vector<lite::types::CustomObjectType> detected_boxes;
    face_detector->detect(frame, detected_boxes);

    // Convert detections to format compatible with utils visualization
    std::vector<lite::types::CustomObjectType> boxes_with_info = detected_boxes;

    // Process each detected face
    for (size_t i = 0; i < detected_boxes.size(); i++) {
        // Extract landmarks for alignment
        std::vector<float> landmarks;
        const auto& face_landmarks = detected_boxes[i].landmarks.points;
        for (const auto& point : face_landmarks) {
            landmarks.push_back(point.x);
            landmarks.push_back(point.y);
        }

        // Align face
        cv::Mat aligned_face = aligner.crop_image_by_mat(frame, landmarks);

        // Get face embedding
        lite::types::FaceContent face_content;
        face_extractor->detect(aligned_face, face_content);

        if (face_content.flag) {
            // Use FaceStorage's match_face function
            auto [identity, similarity] = storage.match_face(
                face_content.embedding,
                face_database,
                similarity_threshold
            );

            // Update box information for visualization
            boxes_with_info[i].className = identity.empty() ? "Unknown" : identity.c_str();
            boxes_with_info[i].confidence = similarity;
            boxes_with_info[i].flag = true;
        }
    }

    return boxes_with_info;
}
bool ANSCustomClass::Destroy()
{
    // Destroy any references
    return true;
}
bool ANSCustomClass::Initialize(const std::string& modelDirectory, std::string& labelMap)
{
    //1. The modelDirectory is supplied by ANSVIS and contains the path to the model files
    _modelDirectory = modelDirectory;

    //2. User can start impelementing the initialization logic here
    std::string face_dectection_model_path = modelDirectory + "/scrfd_10g_bnkps.onnx";
    std::string face_extraction_model_path = modelDirectory + "/adaface_ir_101_webface4m.onnx";
    std::string database_path = "face_database.bin";

    face_detector = new ortcv::SCRFD(detection_model);
    face_extractor = new ortcv::AdaFace(extraction_model);
    AffineAlignment aligner(cv::Size(112, 112));
    float similarity_threshold = 0.4f;


    // Load face database
    FaceStorage storage;
    if (!storage.load(database_path, face_database)) {
        std::cerr << "Error: Could not load face database from " << database_path << std::endl;
    }
    else {
        std::cout << "Loaded face database with " << face_database.size() << " users" << std::endl;
    }

    //3 User also need to return the labelMap which is the name of the class
    // In this example, we will return "CName" as the class name
    labelMap = "face";

    //4. Return true if the initialization is successful
    return true;
}
ANSCustomClass::~ANSCustomClass()
{
    delete face_detector;
    delete face_extractor;
}


// expose the class to the outside world
extern "C" __declspec(dllexport) IANSCustomClass* Create() {
    return new ANSCustomClass();
}