#pragma once
#include "common.h"
#include "llama.h"

#include <string>
#include <unordered_set>
#include <list>
// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"
#include "server-prompt-store.h"

#include <functional>
#include <memory>
#include <vector>

using json = nlohmann::ordered_json;

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};



enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_DECISION,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_SET_LORA,
    SERVER_TASK_TYPE_LOAD_CONTROL_VECTOR,
    SERVER_TASK_TYPE_UNLOAD_CONTROL_VECTOR,
    SERVER_TASK_TYPE_SET_CONTROL_VECTOR,
};

enum oaicompat_type {
    OAICOMPAT_TYPE_NONE,
    OAICOMPAT_TYPE_CHAT,
    OAICOMPAT_TYPE_COMPLETION,
    OAICOMPAT_TYPE_EMBEDDING,
    OAICOMPAT_TYPE_ANTHROPIC,
    OAICOMPAT_TYPE_RESP,
};


struct slot_params {
    bool stream = true;
    bool include_usage = false;
    bool cache_prompt = true; // remember the prompt to avoid reprocessing all prompt

    int32_t  n_keep = 0; // number of tokens to keep from initial prompt
    int32_t  n_discard = 0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t  n_predict = -1; // new tokens to predict

    thinking_tokens think_tokens;

    std::vector<std::string> antiprompt;

    bool timings_per_token = false;
    bool post_sampling_probs = false;
    json input_prefix;
    json input_suffix;

    // speculative decoding parameters
    struct common_params_speculative speculative;

    // OAI-compat fields
    oaicompat_type        oaicompat = OAICOMPAT_TYPE_NONE;
    std::string           oaicompat_model;
    std::string           oaicompat_cmpl_id;
    common_chat_parser_params           chat_parser_params;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)
};

struct server_task {
    int id = -1; // to be filled by server_queue
    int id_multi = -1;

    int index = -1; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot = -1;

    // used by SERVER_TASK_TYPE_INFERENCE
    struct slot_params params;
    server_tokens tokens;

    server_task_type type;
    json data;

    bool metrics_memory = false;

    bool infill = false;
    bool embedding = false;

    // used by SERVER_TASK_TYPE_DECISION
    struct decision {
        std::vector<llama_token> labels; // logits of these tokens, at the last prompt token
    };
    decision decision;

    server_task() = default;
    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task>& tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
        }
        return ids;
    }
};

struct result_timings {
    int32_t prompt_n = -1;
    double prompt_ms;
    double prompt_per_token_ms;
    double prompt_per_second;

    int32_t predicted_n = -1;
    double predicted_ms;
    double predicted_per_token_ms;
    double predicted_per_second;
    int32_t n_ctx = 0;
    int32_t n_past = 0;

    // Optional speculative metrics - only included when > 0
    int32_t draft_n = 0;
    int32_t draft_n_accepted = 0;
    std::vector<int32_t> draft_n_by_depth;
    std::vector<int32_t> draft_n_accepted_by_depth;

    json to_json() const;
};

struct server_task_result {
    int id = -1;
    int id_multi = -1;

    json data;

    stop_type stop = STOP_TYPE_NONE;
    bool error;
    bool final_result = false;
    result_timings timings;
    // OAI-compat fields
    //bool                  verbose = false;
    oaicompat_type        oaicompat = OAICOMPAT_TYPE_NONE;
    std::string           oaicompat_model;
    std::string           oaicompat_cmpl_id;
    common_chat_format    oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_chat_msg    oaicompat_msg;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs;

    int index = 0;

    std::string content;
    std::vector<llama_token> tokens;

    bool stream;
    bool include_usage;
    std::string prompt;
    //slot_params generation_params;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;

    bool post_sampling_probs = false;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    //slot_params generation_params;

    bool                  verbose = false;

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }

    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        // in stream mode, final responses are considered stop
        return true;
    }

    virtual json to_json() {
        return {};
    };

    int get_index() {
        return index;
    }

};

struct server_task_result_cmpl_partial : server_task_result {
    bool anthropic_has_reasoning = false;
    bool anthropic_thinking_block_started = false;
    bool anthropic_text_block_started = false;

    bool oai_resp_thinking_block_started = false;
    bool oai_resp_text_block_started = false;

    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    json to_json_non_oaicompat_partial();

    json to_json_oaicompat_partial();

    json to_json_anthropic_partial();

    json to_json_oaicompat_chat_partial();

    json to_json_oaicompat_resp_partial();

    virtual json to_json() override {
        switch (oaicompat) {
        case OAICOMPAT_TYPE_NONE:
            return to_json_non_oaicompat_partial();
        case OAICOMPAT_TYPE_COMPLETION:
            return to_json_oaicompat_partial();
        case OAICOMPAT_TYPE_CHAT:
            return to_json_oaicompat_chat_partial();
        case OAICOMPAT_TYPE_ANTHROPIC:
            return to_json_anthropic_partial();
        case OAICOMPAT_TYPE_RESP:
            return to_json_oaicompat_resp_partial();
        default:
            GGML_ASSERT(false && "Invalid oaicompat_type");
        };
    }
};

struct server_task_result_cmpl_final : server_task_result {
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    bool anthropic_thinking_block_started = false;
    bool anthropic_text_block_started = false;

    virtual bool is_stop() override {
        return true;
    }

    json to_json_non_oaicompat_final();

    json usage_json_oaicompat();

    json to_json_oaicompat_final();

    json to_json_oaicompat_chat_final();

    json to_json_anthropic_final();

    json to_json_anthropic_stream();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp_final();

    json to_json_oaicompat_resp_stream();

    virtual json to_json() override {
        switch (oaicompat) {
        case OAICOMPAT_TYPE_NONE:
            return to_json_non_oaicompat_final();
        case OAICOMPAT_TYPE_COMPLETION:
            return to_json_oaicompat_final();
        case OAICOMPAT_TYPE_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat_final();
        case OAICOMPAT_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic_final();
        case OAICOMPAT_TYPE_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp_final();
        default:
            GGML_ASSERT(false && "Invalid oaicompat_type");
        }
    }
};

struct server_task_result_error : server_task_result {
    int index = 0;
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override {
        json res = format_error_response(err_msg, err_type);
        if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            res["n_prompt_tokens"] = n_prompt_tokens;
            res["n_ctx"]           = n_ctx;
        }
        return res;
    }
};

struct server_task_result_embd : server_task_result {
    int index = 0;
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // OAI-compat fields
    oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;

    virtual json to_json() override {
        return oaicompat == OAICOMPAT_TYPE_EMBEDDING
            ? to_json_oaicompat()
            : to_json_non_oaicompat();
    }

    json to_json_non_oaicompat() {
        return json{
            {"index",     index},
            {"embedding", embedding},
        };
    }

    json to_json_oaicompat() {
        return json{
            {"index",            index},
            {"embedding",        embedding[0]},
            {"tokens_evaluated", n_tokens},
        };
    }
};

struct server_task_result_decision : server_task_result {
    std::vector<float> scores; // one raw model output per option

    int32_t n_tokens;

    virtual json to_json() override {
        return json{
            {"index",            index},
            {"scores",           scores},
            {"tokens_evaluated", n_tokens},
        };
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct server_prompt_checkpoint {
    llama_pos pos_min;
    llama_pos pos_max;

    int64_t n_tokens;

    // the checkpoint taken at the end of the prompt; kept while another entry can be evicted instead
    bool prompt_end = false;

    std::vector<uint8_t> data;

    // disk-spill location when data was offloaded (empty = resident / no spill)
    std::string spill_path;

    size_t size() const {
        return data.size();
    }

    json to_json() const {
        json j;
        j["pos_min"] = pos_min;
        j["pos_max"] = pos_max;
        j["n_tokens"] = n_tokens;
        return j;
    }

    void from_json(const json & j) {
        pos_min = j.value<llama_pos>("pos_min", 0);
        pos_max = j.value<llama_pos>("pos_max", 0);
        n_tokens = j.value<int64_t>("n_tokens", 0);
    }
};


// What a server was started with, written with every file of the prompt cache. 0: the file does not say (those
// of the versions up to rhel8-b5399 say the context size by their fingerprint, and nothing else).
struct server_prompt_settings {
    uint32_t n_ctx      = 0;
    uint32_t n_ubatch   = 0;
    uint32_t n_parallel = 0;
    // Of this server only, not written: whether the size of the batches and the number of slots are part of how
    // the state of a prompt is laid out (the compacted sliding window; the streams of DeepSeek-V4). A file
    // written with another one is then known not to come back here.
    bool batch_in_state = false;
    bool slots_in_state = false;

    uint32_t n_ctx_slot() const {
        return n_ctx / (n_parallel > 0 ? n_parallel : 1);
    }
    // whether a file written with `other` comes from a server set up in another way, as far as it says
    bool differs(const server_prompt_settings & other) const {
        return (other.n_ctx      != 0 && other.n_ctx      != n_ctx) ||
               (other.n_ubatch   != 0 && other.n_ubatch   != n_ubatch) ||
               (other.n_parallel != 0 && other.n_parallel != n_parallel);
    }
    // whether a file written with `other` is known not to come back in this server
    bool cannot_take(const server_prompt_settings & other) const {
        return (batch_in_state && other.n_ubatch   != 0 && other.n_ubatch   != n_ubatch) ||
               (slots_in_state && other.n_parallel != 0 && other.n_parallel != n_parallel);
    }
};

struct server_prompt {
    server_tokens tokens;
    int n_kept_prompt;
    int n_discarded_prompt;
    thinking_tokens think_tokens;

    std::vector<uint8_t> data;

    std::list<server_prompt_checkpoint> checkpoints;

    // A prompt of a cache that is kept in a directory (--cache-ram-dir): its state and its checkpoints are in
    // this file, data and checkpoints are empty.
    server_prompt_store::entry file;
    // the last position of the state in the file, to check what comes back from it (-1: not known)
    llama_pos file_pos_max = -1;
    // the settings of the server that wrote the file
    server_prompt_settings file_settings;
    // The file did not come back in this server, which is not set up as the one that wrote it: it stays in the
    // directory, where it counts and goes like the others, and is not tried again by this server.
    bool not_here = false;
    // The slot that took this prompt back from its file. The file stays, so that the prompt is not lost if the
    // server dies, until that slot writes a newer state of the same conversation. -1: no slot has it.
    int held_by = -1;
    // times the file could not be read
    int n_failed = 0;

    bool in_file() const {
        return !file.path.empty();
    }

    size_t size() const;

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt{
            tokens.clone(),
            n_kept_prompt,
            n_discarded_prompt,
            think_tokens,
            data,
            checkpoints
        };
    }

    json to_json() const
    {
        json j;
        j["tokens"] = tokens.to_json();
        j["n_kept_prompt"] = n_kept_prompt;
        j["n_discarded_prompt"] = n_discarded_prompt;
        return j;
    }

    void from_json(const json & j) {
        tokens.from_json(j.at("tokens"));
        n_kept_prompt = j.value<llama_pos>("n_kept_prompt", 0);
        n_discarded_prompt = j.value<llama_pos>("n_discarded_prompt", 0);
    }
};

struct server_prompt_cache {
    server_prompt_cache(llama_context* ctx, int32_t limit_size_mib, size_t limit_tokens) {
        this->ctx = ctx;
        this->limit_size = 1024ull * 1024ull * (limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;
    llama_context* ctx;
    size_t size() const;

    size_t n_tokens() const;

    server_prompt* alloc(const server_prompt& prompt, size_t state_size);

    bool load(server_prompt& prompt, const server_tokens& tokens_new, llama_context* ctx, int32_t id_slot, float min_reusable_fraction);

    void update();

    // The cache kept in a directory (--cache-ram-dir), null if it is in the memory of the server.
    // With it, a prompt goes to a file when it is saved (save_to_store, in the place of alloc), is read from the
    // file when it is loaded, and the file goes when the prompt leaves the cache.
    std::unique_ptr<server_prompt_store> store;

    // Opens the directory and takes the prompts that are there for this model. False: the cache stays in memory.
    // A prompt is listed when it can come back here: it fits in the context of a slot, and it was not written
    // with other batches or slots where those are part of the state. The others are left in the directory.
    // `earlier` are fingerprints (server_prompt_store::hash) that an earlier version of the server gave the
    // files of this model, each with the context size it stood for: read and removed like the others.
    bool open_store(const std::string & dir, const std::string & model_fingerprint,
                    const std::vector<std::pair<uint64_t, uint32_t>> & earlier, const server_prompt_settings & settings);
    // the settings of this server, written with every file
    server_prompt_settings store_settings;

    // The prompt of slot id_slot goes to a file. fill gets the place of the state_size bytes of the state and says
    // whether it put them there; pos_max is the last position of that state. When the directory has no room, the
    // older states of this conversation go first, then the oldest prompts, until the file fits.
    bool save_to_store(const server_prompt & prompt, size_t state_size, const std::function<bool(uint8_t *)> & fill,
                       int id_slot, llama_pos pos_max);

    // removes a prompt, with its file
    std::list<server_prompt>::iterator erase(std::list<server_prompt>::iterator it);

    void clear();

private:
    // False if the prompt is in the cache already. The cached prompts that it contains are removed.
    bool forget_contained(const server_prompt & prompt);

    // whether at least half of the cached prompt is the start of the other one, thinking left out as load() does
    bool starts_alike(const server_prompt & cached, const server_prompt & prompt) const;
};
