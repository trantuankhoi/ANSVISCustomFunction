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
};

#endif // FACE_DATA_H