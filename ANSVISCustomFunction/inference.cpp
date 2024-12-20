#include "src/lite.h"
#include <filesystem>
#include <chrono>
#include <thread>
#include <queue>
#include "src/face_database.h"
#include "src/utils.h"
#include <fstream>
#include "nlohmann/json.hpp"


class FaceRecognizer {
private:
    ortcv::SCRFD* face_detector;
    ortcv::AdaFace* face_extractor;
    AffineAlignment aligner;
    std::vector<FaceData> face_database;
    float similarity_threshold = 0.4f;
    FaceStorage storage;
    std::map<int, std::vector<lite::types::CustomObjectType>> logs;
    int c = 0;

public:
    FaceRecognizer(const std::string& detection_model,
        const std::string& extraction_model,
        const std::string& database_path,
        float threshold = 0.4f)
        : aligner(cv::Size(112, 112)), similarity_threshold(threshold) {

        face_detector = new ortcv::SCRFD(detection_model);
        face_extractor = new ortcv::AdaFace(extraction_model);

        // Load face database
        FaceStorage storage;
        if (!storage.load(database_path, face_database)) {
            std::cerr << "Error: Could not load face database from " << database_path << std::endl;
        }
        else {
            std::cout << "Loaded face database with " << face_database.size() << " users" << std::endl;
        }
    }

    ~FaceRecognizer() {
        delete face_detector;
        delete face_extractor;
    }

    std::map<int, std::vector<lite::types::CustomObjectType>> get_logs() {
        return logs;
    }

    cv::Mat processFrame(const cv::Mat& frame) {
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

        // Use utils visualization functions
        cv::Mat visualized = frame.clone();
        lite::utils::draw_boxes_with_landmarks_inplace(visualized, boxes_with_info, true);

        // Logging
        logs[c] = boxes_with_info;

        c++;

        return visualized;
    }
};



void save_log(const std::map<int, std::vector<lite::types::CustomObjectType>>& logs, const std::string& log_path = "log.json") {
    nlohmann::json json_data;

    // Convert each frame's data to JSON
    for (const auto& [frame_number, detections] : logs) {
        nlohmann::json frame_data = nlohmann::json::array();

        for (const auto& detection : detections) {
            nlohmann::json face_data;

            // Basic detection info
            face_data["class_name"] = detection.className;
            face_data["confidence"] = detection.confidence;
            face_data["recognized"] = detection.flag;

            // Bounding box
            face_data["bbox"] = {
                {"x1", detection.box.x},
                {"y1", detection.box.y},
                {"x2", detection.box.x + detection.box.width},
                {"y2", detection.box.y + detection.box.height}
            };

            // Landmarks
            nlohmann::json landmarks = nlohmann::json::array();
            for (const auto& point : detection.landmarks.points) {
                landmarks.push_back({
                    {"x", point.x},
                    {"y", point.y}
                    });
            }
            face_data["landmarks"] = landmarks;

            frame_data.push_back(face_data);
        }

        json_data[std::to_string(frame_number)] = frame_data;
    }

    // Write to file
    std::ofstream output_file(log_path);
    if (output_file.is_open()) {
        output_file << json_data.dump(2);  // Use indent=2 for pretty printing
        output_file.close();
        std::cout << "Successfully saved logs to " << log_path << std::endl;
    }
    else {
        std::cerr << "Error: Could not open file " << log_path << " for writing" << std::endl;
    }
}

void processImages(FaceRecognizer& recognizer, const std::string& input_dir, const std::string& output_dir) {
    std::filesystem::create_directories(output_dir);

    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            cv::Mat frame = cv::imread(entry.path().string());
            if (frame.empty()) continue;

            cv::Mat visualized = recognizer.processFrame(frame);

            // Save results
            std::string output_path = output_dir + "/" + entry.path().filename().string();
            cv::imwrite(output_path, visualized);

            std::cout << "Processed: " << entry.path().filename() << std::endl;
        }
    }

    save_log(recognizer.get_logs());
}

void processVideo(FaceRecognizer& recognizer, const std::string& input_path, const std::string& output_path) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer(output_path,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps, cv::Size(width, height));

    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (cap.read(frame)) {
        cv::Mat visualized = recognizer.processFrame(frame);

        // Calculate and show FPS
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration >= 1) {
            float current_fps = frame_count / duration;
            std::string fps_text = "FPS: " + std::to_string(int(current_fps));
            cv::putText(visualized, fps_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            start_time = current_time;
            frame_count = 0;
        }

        writer.write(visualized);
        //cv::imshow("Processing", visualized);

        if (cv::waitKey(1) == 27) break;  // ESC to stop

        frame_count++;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    save_log(recognizer.get_logs());
}

int main(int argc, char* argv[]) {
    std::string face_detection_path = "models/scrfd_10g_bnkps.onnx";
    std::string face_extraction_path = "models/adaface_ir_101_webface4m.onnx";
    std::string database_path = "face_database.bin";
    std::string input_path = "cam7_checkin_wo_mask.mp4";
    std::string output_path = "result.mp4";

    try {
        FaceRecognizer recognizer(face_detection_path, face_extraction_path, database_path);

        if (std::filesystem::is_directory(input_path)) {
            processImages(recognizer, input_path, output_path);
        }
        else {
            processVideo(recognizer, input_path, output_path);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}