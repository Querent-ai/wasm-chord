#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

// Simple GGUF parser to extract LM head weights
int main() {
    std::string model_path = "/home/puneet/wasm-chord/models/tinyllama-1.1b.Q4_K_M.gguf";
    
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file" << std::endl;
        return 1;
    }
    
    // Read GGUF header
    char magic[4];
    file.read(magic, 4);
    if (strncmp(magic, "GGUF", 4) != 0) {
        std::cerr << "Not a GGUF file" << std::endl;
        return 1;
    }
    
    // Skip to tensor data section (simplified)
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(file_size - 65536000 * 4, std::ios::beg); // Skip to LM head data
    
    // Read LM head weights for token 29892
    size_t token_id = 29892;
    size_t hidden_size = 2048;
    size_t offset = token_id * hidden_size * 4; // 4 bytes per f32
    
    file.seekg(offset, std::ios::cur);
    
    std::vector<float> weights(5);
    file.read(reinterpret_cast<char*>(weights.data()), 5 * sizeof(float));
    
    std::cout << "LM_HEAD[" << token_id << "][0:5]: ";
    for (int i = 0; i < 5; i++) {
        std::cout << weights[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
