#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <ctype.h>
#include <filesystem>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-ngl n_gpu_layers] -embd-mode [prompt]\n", argv[0]);
    printf("\n");
}

// Debug function to print intermediate values
void print_debug_values(const char* stage, const float* values, int count) {
    printf("DEBUG_%s[0:5]: ", stage);
    for (int i = 0; i < count && i < 5; i++) {
        printf("%.6f ", values[i]);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "Hello";
    int ngl = 0;
    bool embedding_mode = false;
    bool debug_mode = false;

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-embd-mode") == 0) {
                if (i + 1 < argc) {
                    try {
                        embedding_mode = true;
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-debug") == 0) {
                debug_mode = true;
            } else {
                // prompt starts here
                break;
            }
        }

        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }

        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // Extract basename from model_path
    const char * basename = strrchr(model_path.c_str(), '/');
    basename = (basename == NULL) ? model_path.c_str() : basename + 1;

    char model_name[256];
    strncpy(model_name, basename, 255);
    model_name[255] = '\0';

    char * dot = strrchr(model_name, '.');
    if (dot != NULL && strcmp(dot, ".gguf") == 0) {
        *dot = '\0';
    }
    printf("Model name: %s\n", model_name);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;
    if (embedding_mode) {
        ctx_params.embeddings = true;
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        ctx_params.n_ubatch = ctx_params.n_batch;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    printf("Input prompt: \"%s\"\n", prompt.c_str());
    printf("Tokenized prompt (%d tokens): ", n_prompt);
    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }
    printf("\n");

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (debug_mode) {
        printf("DEBUG: About to call llama_decode\n");
    }

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return 1;
    }

    if (debug_mode) {
        printf("DEBUG: llama_decode completed\n");
    }

    float * logits;
    int n_logits;
    const char * type;

    if (embedding_mode) {
        logits = llama_get_embeddings(ctx);
        n_logits = llama_n_embd(model);
        type = "embeddings";
    } else {
        logits = llama_get_logits(ctx);
        n_logits = llama_n_vocab(model);
        type = "logits";
    }

    printf("Vocab size: %d\n", llama_n_vocab(model));
    printf("Saving %s to data/llamacpp-tinyllama-1.1b.Q4_K_M.bin\n", type);
    printf("First 10 %s: ", type);
    for (int i = 0; i < 10 && i < n_logits; i++) {
        printf("%.6f ", logits[i]);
    }
    printf("\nLast 10 %s: ", type);
    for (int i = n_logits - 10; i < n_logits; i++) {
        printf("%.6f ", logits[i]);
    }
    printf("\n");

    // Save to binary file
    std::filesystem::create_directories("data");
    FILE * f = fopen("data/llamacpp-tinyllama-1.1b.Q4_K_M.bin", "wb");
    if (f) {
        fwrite(logits, sizeof(float), n_logits, f);
        fclose(f);
    }

    // Save to text file
    FILE * f_txt = fopen("data/llamacpp-tinyllama-1.1b.Q4_K_M.txt", "w");
    if (f_txt) {
        for (int i = 0; i < n_logits; i++) {
            fprintf(f_txt, "%.6f\n", logits[i]);
        }
        fclose(f_txt);
    }

    printf("\n%s saved to data/llamacpp-tinyllama-1.1b.Q4_K_M.bin\n", type);
    printf("%s saved to data/llamacpp-tinyllama-1.1b.Q4_K_M.txt\n", type);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
