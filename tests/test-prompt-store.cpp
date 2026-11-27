// The files of the prompt cache that the server keeps in a directory (--cache-ram-dir,
// examples/server/server-prompt-store.h): what is written is what is read back, by the same process and by one
// that starts later; one store at a time has the directory; the files of another model are left alone and no
// file is ever replaced; a file that was not finished, or cannot be written for lack of room, leaves nothing
// behind. No model is needed: the state is random bytes here.

#include "../examples/server/server-prompt-store.h"

#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#if SERVER_PROMPT_STORE_POSIX

#include <csignal>
#include <sys/resource.h>

static int n_fail = 0, n_checks = 0;

static void check(bool ok, const char * what) {
    ++n_checks;
    if (!ok) ++n_fail;
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
}

static std::mt19937 rng(34);

static std::vector<uint8_t> bytes(size_t n) {
    std::vector<uint8_t> v(n);
    for (auto & b : v) b = (uint8_t)rng();
    return v;
}

struct conversation {
    std::string meta;
    std::vector<uint8_t> state;
    std::vector<server_prompt_store::ckpt_out> ckpts;
    server_prompt_store::entry entry;
};

static conversation make(size_t state_size, int n_ckpt) {
    conversation c;
    c.meta = "{\"tokens\":[" + std::to_string(rng()) + "],\"n\":" + std::to_string(state_size) + "}";
    c.state = bytes(state_size);
    for (int i = 0; i < n_ckpt; ++i) {
        server_prompt_store::ckpt_out k;
        k.pos_min = (int32_t)(rng()%1000); k.pos_max = k.pos_min + (int32_t)(rng()%100000);
        k.n_tokens = (int64_t)rng()*7; k.prompt_end = i == n_ckpt - 2;
        k.data = bytes(i == 1 ? 0 : 1 + rng()%70000);
        c.ckpts.push_back(std::move(k));
    }
    return c;
}

static bool write(server_prompt_store & store, conversation & c, bool fill_ok = true) {
    std::vector<server_prompt_store::ckpt_in> in;
    for (const auto & k : c.ckpts) in.push_back({ k.pos_min, k.pos_max, k.n_tokens, k.prompt_end, k.data.data(), k.data.size() });
    return store.write(c.meta, c.state.size(), [&] (uint8_t * dst) {
        if (fill_ok) memcpy(dst, c.state.data(), c.state.size());
        return fill_ok;
    }, in, c.entry);
}

// what the store gives for an entry is what the conversation was made of
static bool same(server_prompt_store & store, const server_prompt_store::entry & e, const conversation & c) {
    if (e.meta != c.meta || e.state_size != c.state.size() || e.n_ckpt != c.ckpts.size()) return false;
    if (e.state_size > 0) {
        const uint8_t * p = store.map_state(e);
        const bool ok = p && memcmp(p, c.state.data(), c.state.size()) == 0;
        store.unmap_state(e, p);
        if (!ok) return false;
    }
    std::vector<server_prompt_store::ckpt_out> got;
    if (!store.read_checkpoints(e, got) || got.size() != c.ckpts.size()) return false;
    size_t total = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const auto & a = got[i]; const auto & b = c.ckpts[i];
        if (a.pos_min != b.pos_min || a.pos_max != b.pos_max || a.n_tokens != b.n_tokens || a.prompt_end != b.prompt_end || a.data != b.data) return false;
        total += a.data.size();
    }
    return total == e.ckpt_size;
}

static size_t count_files(const std::string & dir) {
    size_t n = 0;
    if (DIR * d = opendir(dir.c_str())) {
        while (const dirent * de = readdir(d)) n += de->d_name[0] != '.';
        closedir(d);
    }
    return n;
}

int main() {
    const char * tmp = getenv("TMPDIR");
    std::string base = std::string(tmp && tmp[0] ? tmp : "/tmp") + "/test-prompt-store-XXXXXX";
    if (!mkdtemp(&base[0])) { printf("cannot make a directory in %s\n", tmp && tmp[0] ? tmp : "/tmp"); return 1; }
    const std::string dir = base + "/a/cache";
    const uint64_t model = server_prompt_store::hash("model A|1234|5120");
    const uint64_t model_b = server_prompt_store::hash("model B|1234|5120");
    struct stat st;
    bool ok;

    // conversations of several sizes: a state of one byte, one that ends on a page, large ones, no checkpoint or several
    std::vector<conversation> all;
    all.push_back(make(1, 0));
    all.push_back(make(4096, 1));
    all.push_back(make(5000, 3));
    all.push_back(make((size_t)3 << 20, 5));
    all.push_back(make(((size_t)1 << 20) + 17, 32));
    {
        server_prompt_store store;
        check(store.open(dir + "/", model), "the directory is made, with the ones above it");
        check(stat(dir.c_str(), &st) == 0 && (st.st_mode & 0777) == 0700, "it is for its owner only");
        ok = true;
        for (auto & c : all) ok = write(store, c) && ok;
        check(ok, "five conversations are written");
        check(count_files(dir) == all.size(), "one file each, and nothing else to see in the directory");
        ok = true;
        for (auto & c : all) ok = same(store, c.entry, c) && ok;
        check(ok, "each one reads back as it was written");
        ok = true;
        for (auto & c : all) ok = stat(c.entry.path.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600 && ok;
        check(ok, "the files are for their owner only");

        // one server at a time
        server_prompt_store second;
        check(!second.open(dir, model) && !second.error.empty(), "a second store cannot open the directory while the first one has it");
        printf("     (the reason given: %s)\n", second.error.c_str());
    }

    // a process that starts later
    {
        server_prompt_store later;
        size_t n_other = 9, n_bad = 9;
        ok = later.open(dir, model);
        auto found = later.scan(n_other, n_bad);
        ok = ok && found.size() == all.size() && n_other == 0 && n_bad == 0;
        for (size_t i = 0; ok && i < found.size(); ++i) ok = found[i].seq == all[i].entry.seq && same(later, found[i], all[i]);
        check(ok, "a store opened once the first one is closed finds them, the oldest first, and reads the same bytes");

        // a file that sits where the next one would go is not replaced
        char name[80];
        snprintf(name, sizeof(name), "/p-%016llx-%016llx.ikpc", (unsigned long long)model, (unsigned long long)later.next_seq);
        const std::string taken = dir + name;
        FILE * f = fopen(taken.c_str(), "wb"); fputs("someone else's", f); fclose(f);
        conversation c = make(777, 2);
        ok = write(later, c) && c.entry.seq > all.back().entry.seq && c.entry.path != taken && same(later, c.entry, c);
        char buf[32] = { 0 };
        f = fopen(taken.c_str(), "rb"); const size_t n = fread(buf, 1, sizeof(buf) - 1, f); fclose(f);
        check(ok && n == 14 && strcmp(buf, "someone else's") == 0, "what it writes then is newer than what it found, and takes the place of no file");
        unlink(taken.c_str());
        later.remove(c.entry);
        check(count_files(dir) == all.size(), "a conversation that is removed takes its file with it");
    }

    // another model in the same directory
    {
        server_prompt_store other;
        size_t n_other = 0, n_bad = 0;
        ok = other.open(dir, model_b);
        auto found = other.scan(n_other, n_bad);
        check(ok && found.empty() && n_other == all.size() && n_bad == 0 && count_files(dir) == all.size(),
              "another model finds none of them and leaves the files");
        conversation c = make(3000, 1);
        ok = write(other, c) && same(other, c.entry, c) && count_files(dir) == all.size() + 1;
        for (auto & a : all) ok = ok && c.entry.path != a.entry.path;
        check(ok, "what the other model writes goes to a file of its own");
        other.remove(c.entry);
    }
    {
        server_prompt_store again;
        size_t n_other = 0, n_bad = 0;
        ok = again.open(dir, model);
        auto found = again.scan(n_other, n_bad);
        ok = ok && found.size() == all.size();
        for (size_t i = 0; ok && i < found.size(); ++i) ok = same(again, found[i], all[i]);
        check(ok, "and the conversations of the first model are as they were");
    }

    // files that are not conversations, or not whole
    {
        const std::string unfinished = dir + "/p-00000000000000aa.ikpc.tmp", cut = dir + "/p-00000000000000ab.ikpc",
                          junk = dir + "/p-00000000000000ac.ikpc", foreign = dir + "/notes.txt";
        FILE * f = fopen(unfinished.c_str(), "wb"); fputs("half", f); fclose(f);
        f = fopen(junk.c_str(), "wb"); for (int i = 0; i < 300; ++i) fputc(i, f); fclose(f);
        f = fopen(foreign.c_str(), "wb"); fputs("not ours", f); fclose(f);
        // a copy of a good file without its end
        {
            FILE * in = fopen(all[2].entry.path.c_str(), "rb"); f = fopen(cut.c_str(), "wb");
            std::vector<char> buf(4500);
            const size_t n = fread(buf.data(), 1, buf.size(), in);
            fwrite(buf.data(), 1, n, f); fclose(in); fclose(f);
        }
        server_prompt_store later;
        size_t n_other = 0, n_bad = 0;
        later.open(dir, model);
        auto found = later.scan(n_other, n_bad);
        check(found.size() == all.size() && n_bad == 2 && n_other == 0, "a file cut short and one of noise are not taken for conversations");
        check(access(unfinished.c_str(), F_OK) != 0, "a file that was never finished is removed");
        check(access(foreign.c_str(), F_OK) == 0 && access(cut.c_str(), F_OK) == 0, "what the store did not write, or cannot read, is left where it is");
        unlink(cut.c_str()); unlink(junk.c_str()); unlink(foreign.c_str());
    }

    // a directory that others can write in is refused
    {
        const std::string open_dir = base + "/open";
        mkdir(open_dir.c_str(), 0700);
        chmod(open_dir.c_str(), 0777);
        server_prompt_store s;
        check(!s.open(open_dir, model) && !s.error.empty(), "a directory that other accounts can write in is refused");
        printf("     (the reason given: %s)\n", s.error.c_str());
        chmod(open_dir.c_str(), 0700);
        unlink((open_dir + "/.lock").c_str());
        rmdir(open_dir.c_str());
    }

    // a state that cannot be copied, and a directory without room: nothing is left, and the store goes on
    {
        server_prompt_store store;
        size_t n_other = 0, n_bad = 0;
        store.open(dir, model);
        store.scan(n_other, n_bad);
        conversation c = make(100000, 2);
        const bool written = write(store, c, false);
        check(!written && !store.error.empty() && !store.no_room && count_files(dir) == all.size(), "a state that could not be copied leaves no file");

        struct rlimit old_limit, small = { 1 << 20, RLIM_INFINITY };
        getrlimit(RLIMIT_FSIZE, &old_limit);
        small.rlim_max = old_limit.rlim_max;
        signal(SIGXFSZ, SIG_IGN);
        setrlimit(RLIMIT_FSIZE, &small);
        conversation big = make((size_t)4 << 20, 1);
        const bool written_big = write(store, big);
        setrlimit(RLIMIT_FSIZE, &old_limit);
        check(!written_big && store.no_room && count_files(dir) == all.size(), "no room for a file: the write fails before it starts, says so, and leaves nothing");
        printf("     (the reason given: %s)\n", store.error.c_str());

        conversation after = make(2000, 1);
        check(write(store, after) && !store.no_room && same(store, after.entry, after), "the next conversation is written as usual");
        store.remove(after.entry);

        for (auto & a : all) store.remove(a.entry);
        check(count_files(dir) == 0, "all removed: the directory is empty");
    }
    unlink((dir + "/.lock").c_str());
    rmdir(dir.c_str()); rmdir((base + "/a").c_str()); rmdir(base.c_str());

    printf("%d of %d checks failed\n", n_fail, n_checks);
    return n_fail ? 1 : 0;
}

#else

int main() {
    printf("the prompt cache is not kept in a directory on this system: nothing to check\n");
    return 0;
}

#endif
