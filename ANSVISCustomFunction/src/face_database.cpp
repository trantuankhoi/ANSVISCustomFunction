#include "face_database.h"
#include <fstream>

bool FaceStorage::save(const std::string& filename, const std::vector<FaceData>& database) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    size_t num_users = database.size();
    file.write(reinterpret_cast<const char*>(&num_users), sizeof(num_users));

    for (const auto& user : database) {
        size_t name_length = user.username.length();
        file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
        file.write(user.username.c_str(), name_length);

        size_t num_embeddings = user.embeddings.size();
        file.write(reinterpret_cast<const char*>(&num_embeddings), sizeof(num_embeddings));

        for (const auto& embedding : user.embeddings) {
            size_t embedding_size = embedding.size();
            file.write(reinterpret_cast<const char*>(&embedding_size), sizeof(embedding_size));
            file.write(reinterpret_cast<const char*>(embedding.data()), embedding_size * sizeof(float));
        }
    }
    return true;
}

bool FaceStorage::load(const std::string& filename, std::vector<FaceData>& database) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    size_t num_users;
    file.read(reinterpret_cast<char*>(&num_users), sizeof(num_users));
    database.resize(num_users);

    for (auto& user : database) {
        size_t name_length;
        file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
        user.username.resize(name_length);
        file.read(&user.username[0], name_length);

        size_t num_embeddings;
        file.read(reinterpret_cast<char*>(&num_embeddings), sizeof(num_embeddings));
        user.embeddings.resize(num_embeddings);

        for (auto& embedding : user.embeddings) {
            size_t embedding_size;
            file.read(reinterpret_cast<char*>(&embedding_size), sizeof(embedding_size));
            embedding.resize(embedding_size);
            file.read(reinterpret_cast<char*>(embedding.data()), embedding_size * sizeof(float));
        }
    }
    return true;
}

float FaceStorage::cosine_similarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size() || vec1.empty()) {
        return -1.0f;
    }

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < vec1.size(); i++) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    if (norm1 == 0.0f || norm2 == 0.0f) {
        return -1.0f;
    }

    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

std::pair<std::string, float> FaceStorage::match_face(const std::vector<float>& query_embedding,
    const std::vector<FaceData>& database,
    float similarity_threshold) {
    float best_similarity = -1.0f;
    std::string best_match = "";

    for (const auto& user : database) {
        for (const auto& stored_embedding : user.embeddings) {
            float similarity = cosine_similarity(query_embedding, stored_embedding);

            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_match = user.username;
            }
        }
    }

    // Return empty string if similarity is below threshold
    if (best_similarity < similarity_threshold) {
        return std::make_pair("", best_similarity);
    }

    return std::make_pair(best_match, best_similarity);
}