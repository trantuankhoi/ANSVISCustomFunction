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