#include "src/lite.h"

int main(int argc, char* argv[]) {
	std::string onnx_path = "D:\\FTECH\\01_Project\\SVL\\ANSVISCustomFunction\\ANSVISCustomFunction\\scrfd_10g_bnkps.onnx";
	std::string test_img_path = "D:\\FTECH\\01_Project\\SVL\\ANSVISCustomFunction\\ANSVISCustomFunction\\largest_selfie.jpg";
	std::string save_img_path = "D:\\FTECH\\01_Project\\SVL\\ANSVISCustomFunction\\ANSVISCustomFunction\\test_results.jpg";

	auto* face_detector = new ortcv::SCRFD(onnx_path);
	std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
	cv::Mat img_bgr = cv::imread(test_img_path);
	face_detector->detect(img_bgr, detected_boxes);

	cv::Mat visualized = lite::utils::draw_boxes_with_landmarks(img_bgr, detected_boxes);
	cv::imwrite(save_img_path, visualized);
	delete face_detector;
	return 0;
}