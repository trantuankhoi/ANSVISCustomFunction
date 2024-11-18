#ifndef FACE_DATA_H
#define FACE_DATA_H

#include <string>
#include <vector>

struct FaceData {
    std::string username;
    std::vector<std::vector<float>> embeddings;
};

class FaceStorage {
public:
    static bool save(const std::string& filename, const std::vector<FaceData>& database);
    static bool load(const std::string& filename, std::vector<FaceData>& database);
	static float cosine_similarity(const std::vector<float>& vec1, const std::vector<float>& vec2);
    std::pair<std::string, float> match_face(const std::vector<float>& query_embedding, const std::vector<FaceData>& database, float similarity_threshold);
};

#endif // FACE_DATA_H