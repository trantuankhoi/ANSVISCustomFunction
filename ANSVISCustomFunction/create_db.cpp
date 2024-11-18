#include "src/lite.h"
#include "src/face_database.h"
#include <filesystem>


int creat_db(int argc, char* argv[]) {
    std::string face_extraction_path = "adaface_ir_101_webface4m.onnx";
    std::string root_dir = "facedb";
    std::string output_file = "face_database.bin";

    ortcv::AdaFace face_extractor(face_extraction_path);

    std::vector<FaceData> database;

    for (const auto& user_dir : std::filesystem::directory_iterator(root_dir)) {
        if (!std::filesystem::is_directory(user_dir)) continue;

        FaceData user_data;
        user_data.username = user_dir.path().filename().string();

        for (const auto& img_path : std::filesystem::directory_iterator(user_dir)) {
            if (img_path.path().extension().string().find(".jpg") == std::string::npos) continue;
            std::cout << "processing: " << img_path << "\n";

            cv::Mat img_bgr = cv::imread(img_path.path().string());

            lite::types::FaceContent face_content;
            face_extractor.detect(img_bgr, face_content);

            if (face_content.flag) {
                user_data.embeddings.push_back(face_content.embedding);
            }
        }

        if (!user_data.embeddings.empty()) {
            database.push_back(user_data);
        }
    }

    FaceStorage::save(output_file, database);
    return 0;
}