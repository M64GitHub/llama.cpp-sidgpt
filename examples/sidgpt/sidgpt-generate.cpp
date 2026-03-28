// sidgpt-generate: SID-GPT inference via llama.cpp
//
// Generates chiptune music as uint16 LE binary tokens,
// playable by sidgpt-play.

#include "llama.h"
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int TOK_SEP   = 256;
static const int TOK_FRAME = 257;
static const int FRAME_SIZE = 26;

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s -m model.gguf [options]\n"
        "\n"
        "Options:\n"
        "  -m PATH          Model file (required)\n"
        "  -n N             Frames to generate (default: 500)\n"
        "  --temp F         Temperature (default: 0.85)\n"
        "  --top-k N        Top-K sampling (default: 0=off)\n"
        "  --top-p F        Top-P sampling (default: 1.0)\n"
        "  --seed N         RNG seed (default: random)\n"
        "  --context-keep N Keep N%% of context on overflow\n"
        "                   (25/50/75, default: 0=off)\n"
        "  --seed-file PATH Prompt from .bin/.sidgpt file\n"
        "  --seed-frames N  Frames to extract (default: 10)\n"
        "  -o PATH          Output file (default: stdout)\n"
        "  -ngl N           GPU layers (default: 99)\n"
        "\n"
        "Output: uint16 LE binary, pipe to sidgpt-play\n",
        prog);
}

// Load seed frames from a binary file (uint16 LE).
// Matches Zig sidgpt behavior: skip leading SEPs,
// extract complete data frames (FRAME + 25 registers).
static std::vector<llama_token> load_seed_frames(
        const char * path, int max_frames) {
    std::vector<llama_token> tokens;

    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr,
            "[sidgpt] cannot open seed file: %s\n",
            path);
        return tokens;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    int n_tokens = (int)(file_size / 2);
    std::vector<uint16_t> raw(n_tokens);
    size_t nr = fread(raw.data(), 2, n_tokens, f);
    n_tokens = (int)nr;
    fclose(f);

    // skip leading SEP tokens
    int pos = 0;
    while (pos < n_tokens && raw[pos] == TOK_SEP)
        pos++;

    // extract data frames
    int added = 0;
    while (pos < n_tokens && added < max_frames) {
        if (raw[pos] == TOK_SEP) {
            pos += FRAME_SIZE; // skip SEP frame
            continue;
        }
        if (raw[pos] == TOK_FRAME) {
            int end = pos + FRAME_SIZE;
            if (end <= n_tokens) {
                for (int p = pos; p < end; p++)
                    tokens.push_back((llama_token)raw[p]);
                added++;
            }
            pos = end;
        } else {
            pos++;
        }
    }

    fprintf(stderr,
        "[sidgpt] loaded %d seed frames (%zu tokens)"
        " from %s\n",
        added, tokens.size(), path);
    return tokens;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string output_path;
    std::string seed_file;
    int n_frames     = 500;
    float temp       = 0.85f;
    int top_k        = 0;
    float top_p      = 1.0f;
    uint32_t seed    = LLAMA_DEFAULT_SEED;
    int context_keep = 0;
    int seed_frames  = 10;
    int ngl          = 99;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i+1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-n") && i+1 < argc) {
            n_frames = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--temp") && i+1 < argc) {
            temp = std::atof(argv[++i]);
        } else if (!strcmp(argv[i], "--top-k") && i+1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--top-p") && i+1 < argc) {
            top_p = std::atof(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i+1 < argc) {
            seed = (uint32_t)std::atol(argv[++i]);
        } else if (!strcmp(argv[i], "--context-keep") && i+1 < argc) {
            context_keep = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--seed-file") && i+1 < argc) {
            seed_file = argv[++i];
        } else if (!strcmp(argv[i], "--seed-frames") && i+1 < argc) {
            seed_frames = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-o") && i+1 < argc) {
            output_path = argv[++i];
        } else if (!strcmp(argv[i], "-ngl") && i+1 < argc) {
            ngl = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-h") ||
                   !strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // validate context-keep
    if (context_keep != 0 && context_keep != 25 &&
            context_keep != 50 && context_keep != 75) {
        fprintf(stderr,
            "[sidgpt] invalid context-keep %d, "
            "must be 0/25/50/75\n", context_keep);
        return 1;
    }

    // build prompt
    std::vector<llama_token> prompt;
    if (!seed_file.empty()) {
        // seed-file: extracted frames only (no SEP prefix)
        prompt = load_seed_frames(
            seed_file.c_str(), seed_frames);
        if (prompt.empty()) {
            fprintf(stderr,
                "[sidgpt] no frames extracted from "
                "seed file\n");
            return 1;
        }
    } else {
        // unconditional: SEP frame + FRAME token
        prompt.assign(FRAME_SIZE, TOK_SEP);
        prompt.push_back(TOK_FRAME);
    }

    const int n_prompt = (int)prompt.size();
    const int n_gen    = n_frames * FRAME_SIZE - 1;
    const int n_total  = n_prompt + n_gen;

    fprintf(stderr,
        "[sidgpt] frames=%d prompt=%d gen=%d temp=%.2f "
        "top_k=%d top_p=%.2f seed=%u ctx_keep=%d%%\n",
        n_frames, n_prompt, n_gen, temp,
        top_k, top_p, seed, context_keep);

    // load backends
    ggml_backend_load_all();

    // load model
    llama_model_params model_params =
        llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model =
        llama_model_load_from_file(
            model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "[sidgpt] failed to load model\n");
        return 1;
    }

    // context size: use model's training ctx when
    // context-keep is active, else just what we need
    int n_ctx;
    if (context_keep > 0) {
        n_ctx = llama_model_n_ctx_train(model);
    } else {
        n_ctx = n_total + 1;
    }

    // create context
    llama_context_params ctx_params =
        llama_context_default_params();
    ctx_params.n_ctx   = n_ctx;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;

    llama_context * ctx =
        llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr,
            "[sidgpt] failed to create context\n");
        return 1;
    }

    fprintf(stderr, "[sidgpt] n_ctx=%d (train=%d)\n",
        llama_n_ctx(ctx),
        llama_model_n_ctx_train(model));

    // create sampler chain
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl =
        llama_sampler_chain_init(sparams);

    if (temp > 0.0f) {
        // Standard llama.cpp order:
        // top_k -> top_p -> temp -> dist
        if (top_k > 0) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_top_k(top_k));
        }
        if (top_p < 1.0f) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_top_p(top_p, 1));
        }
        llama_sampler_chain_add(smpl,
            llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl,
            llama_sampler_init_dist(seed));
    } else {
        llama_sampler_chain_add(smpl,
            llama_sampler_init_greedy());
    }

    // evaluate prompt
    llama_batch batch =
        llama_batch_get_one(prompt.data(), prompt.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "[sidgpt] prompt eval failed\n");
        return 1;
    }

    fprintf(stderr,
        "[sidgpt] prompt evaluated (%d tokens)"
        ", generating...\n", n_prompt);

    // generate tokens
    std::vector<uint16_t> output;
    const auto t_start = ggml_time_us();
    int n_decode = 0;
    int n_slides = 0;

    for (int i = 0; i < n_gen; i++) {
        // context-keep: sliding window
        if (context_keep > 0) {
            llama_memory_t mem = llama_get_memory(ctx);
            llama_pos pos_max =
                llama_memory_seq_pos_max(mem, 0);
            int ctx_size = llama_n_ctx(ctx);

            if (pos_max >= ctx_size - 1) {
                int keep_len =
                    (int)(ctx_size *
                        (context_keep / 100.0f));
                keep_len = (keep_len / FRAME_SIZE) *
                    FRAME_SIZE;
                int drop = pos_max + 1 - keep_len;
                int rem = drop % FRAME_SIZE;
                if (rem != 0)
                    drop += FRAME_SIZE - rem;

                llama_memory_seq_rm(mem, 0, 0, drop);
                llama_memory_seq_add(
                    mem, 0, drop, -1, -drop);

                n_slides++;
                fprintf(stderr,
                    "[sidgpt] [SLIDE] #%d at tok %d, "
                    "kept %d\n",
                    n_slides, n_prompt + i,
                    pos_max + 1 - drop);
            }
        }

        llama_token tok =
            llama_sampler_sample(smpl, ctx, -1);
        output.push_back((uint16_t)tok);

        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr,
                "[sidgpt] decode failed at token %d\n",
                i);
            return 1;
        }
        n_decode++;

        if ((i + 1) % (FRAME_SIZE * 100) == 0) {
            fprintf(stderr,
                "[sidgpt] %d/%d frames\n",
                (i + 1) / FRAME_SIZE, n_frames);
        }
    }

    const auto t_end = ggml_time_us();
    const float t_sec =
        (t_end - t_start) / 1000000.0f;

    fprintf(stderr,
        "[sidgpt] generated %d tokens in %.2f s "
        "(%.1f tok/s)",
        n_decode, t_sec, n_decode / t_sec);
    if (n_slides > 0)
        fprintf(stderr, ", %d slides", n_slides);
    fprintf(stderr, "\n");

    // write output as uint16 LE binary
    FILE * fout = stdout;
    if (!output_path.empty()) {
        fout = fopen(output_path.c_str(), "wb");
        if (!fout) {
            fprintf(stderr,
                "[sidgpt] cannot open %s for writing\n",
                output_path.c_str());
            return 1;
        }
    }

    // write prompt tokens then generated tokens
    for (auto t : prompt) {
        uint16_t tok = (uint16_t)t;
        fwrite(&tok, sizeof(uint16_t), 1, fout);
    }
    fwrite(output.data(), sizeof(uint16_t),
        output.size(), fout);

    if (fout != stdout) {
        fclose(fout);
        fprintf(stderr,
            "[sidgpt] written to %s (%zu bytes)\n",
            output_path.c_str(),
            (prompt.size() + output.size()) *
                sizeof(uint16_t));
    }

    // perf stats
    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
