#include "src/lite.h"
#include <filesystem>


int main1(int argc, char* argv[]) {
	std::string face_detection_path = "D:\\FTECH\\01_Project\\SVL\\ANSVISCustomFunction\\ANSVISCustomFunction\\scrfd_10g_bnkps.onnx";
	std::string face_extraction_path = "D:\\FTECH\\01_Project\\SVL\\ANSVISCustomFunction\\ANSVISCustomFunction\\adaface_ir_101_webface4m.onnx";
	std::string test_img_path = "D:\\FTECH\\01_Project\\SVL\\ANSVISCustomFunction\\ANSVISCustomFunction\\largest_selfie.jpg";
	std::string save_img_path = "D:\\FTECH\\01_Project\\SVL\\ANSVISCustomFunction\\ANSVISCustomFunction\\test_results.jpg";
	std::string save_dir = "D:\\FTECH\\01_Project\\SVL\\ANSVISCustomFunction\\ANSVISCustomFunction\\test_results";

	auto* face_detector = new ortcv::SCRFD(face_detection_path);
	AffineAlignment alinger(cv::Size(112, 112));
	ortcv::AdaFace face_extractor(face_extraction_path);
	std::vector<ortcv::types::FaceContent> face_embeddings;


	if (std::filesystem::exists(save_dir)) {
		std::filesystem::remove_all(save_dir);
	}
	std::filesystem::create_directories(save_dir);

	std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
	cv::Mat img_bgr = cv::imread(test_img_path);
	face_detector->detect(img_bgr, detected_boxes);

	cv::Mat visualized = lite::utils::draw_boxes_with_landmarks(img_bgr, detected_boxes);
	cv::imwrite(save_img_path, visualized);

	for (size_t i = 0; i < detected_boxes.size(); i++) {
		// crop face
		cv::Mat bbox(1, 4, CV_32F);
		bbox.at<float>(0, 0) = detected_boxes[i].box.x1;
		bbox.at<float>(0, 1) = detected_boxes[i].box.y1;
		bbox.at<float>(0, 2) = detected_boxes[i].box.x2;
		bbox.at<float>(0, 3) = detected_boxes[i].box.y2;

		cv::Mat cropped = alinger.crop_image_by_bbox(img_bgr, bbox);

		std::string crop_path = save_dir + "/face_" + std::to_string(i) + "_crop.jpg";
		cv::imwrite(crop_path, cropped);

		// align face
		std::vector<float> landmarks;
		const auto& face_landmarks = detected_boxes[i].landmarks.points;
		for (const auto& point : face_landmarks) {
			landmarks.push_back(point.x);
			landmarks.push_back(point.y);
		}

		cv::Mat aligned_face = alinger.crop_image_by_mat(img_bgr, landmarks);
		std::string align_path = save_dir + "/face_" + std::to_string(i) + "_aligned.jpg";
		cv::imwrite(align_path, aligned_face);

		// extract face embedding
		lite::types::FaceContent face_content;
		face_extractor.detect(aligned_face, face_content);

		if (face_content.flag) {
			std::string embedding_path = save_dir + "/face_" + std::to_string(i) + "_embedding.txt";
			std::ofstream emb_file(embedding_path);
			if (emb_file.is_open()) {
				emb_file << "Dimension: " << face_content.dim << "\n";
				emb_file << "Embedding:\n";
				for (const auto& val : face_content.embedding) {
					emb_file << val << " ";
				}
				emb_file.close();
			}

			face_embeddings.push_back(face_content);
		}
	}

	delete face_detector;
	return 0;
}