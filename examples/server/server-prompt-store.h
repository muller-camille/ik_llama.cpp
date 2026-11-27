// The prompt cache of the server kept as files in a directory (--cache-ram-dir), for a directory on a RAM disk:
// a conversation that is set aside is written to a file there and is not held in the memory of the server,
// and it is still there when the server starts again.
//
// One file per conversation, p-<model>-<number>.ikpc:
//   a header, the description of the prompt (text, the caller's), the state of the sequence, its context
//   checkpoints, an end mark.
// A file is written under another name and gets its name once it is complete, never in the place of another
// file; the room for it is reserved before anything is written, so a full RAM disk makes the write fail at its
// start and nothing else. One server at a time uses a directory (a lock file), and it takes only files that
// belong to its own account.
//
// Nothing here knows about the model: the state is bytes that the caller fills in and reads.

#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#define SERVER_PROMPT_STORE_POSIX 1
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define SERVER_PROMPT_STORE_POSIX 0
#endif

struct server_prompt_store {
    // a context checkpoint to write: the bytes are the caller's
    struct ckpt_in {
        int32_t pos_min, pos_max;
        int64_t n_tokens;
        bool    prompt_end;
        const uint8_t * data;
        size_t  size;
    };
    // a context checkpoint read back
    struct ckpt_out {
        int32_t pos_min, pos_max;
        int64_t n_tokens;
        bool    prompt_end;
        std::vector<uint8_t> data;
    };
    // a conversation in the directory
    struct entry {
        std::string path;
        std::string meta;        // the description the caller gave
        uint64_t seq        = 0; // order of writing: the larger, the newer
        uint64_t state_off  = 0;
        uint64_t state_size = 0;
        uint64_t ckpt_size  = 0; // bytes of the checkpoints, without their headers
        uint64_t n_ckpt     = 0;
        uint64_t fingerprint = 0; // the one in the file: this model's, or one of `also_read`
        int64_t  written_ns  = 0; // the date of the file, in ns
    };

    std::string dir;
    uint64_t    fingerprint = 0; // of the model: files with another one are left alone
    // Other fingerprints whose files are this model's too: read and removed like the others, never written.
    // (An earlier version of the server made the fingerprint of more than the model.)
    std::vector<uint64_t> also_read;
    uint64_t    next_seq    = 1;
    std::string error;           // why the last call failed
    bool        no_room = false; // the last write failed for lack of room in the directory

    static constexpr uint32_t k_magic   = 0x43504b49; // "IKPC"
    static constexpr uint32_t k_end     = 0x444e454b; // "KEND"
    static constexpr uint32_t k_version = 1;

    struct header {
        uint32_t magic, version;
        uint64_t fingerprint, seq, meta_size, state_off, state_size, n_ckpt, ckpt_size, total_size;
    };
    struct ckpt_record {
        int32_t  pos_min, pos_max;
        int64_t  n_tokens;
        uint64_t size;
        uint8_t  prompt_end;
        uint8_t  pad[7];
    };

    static uint64_t hash(const std::string & s) {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
        return h;
    }

    server_prompt_store() = default;
    server_prompt_store(const server_prompt_store &) = delete;
    server_prompt_store & operator=(const server_prompt_store &) = delete;

#if SERVER_PROMPT_STORE_POSIX

    ~server_prompt_store() {
        if (lock_fd >= 0) ::close(lock_fd); // the lock goes with it
    }

    // The directory is made if it is not there, for this account only: the files hold conversations. It has to
    // belong to this account, with nobody else allowed to write in it, and no other server may be using it.
    bool open(const std::string & directory, uint64_t model_fingerprint, const std::vector<uint64_t> & also = {}) {
        begin();
        dir = directory;
        while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
        fingerprint = model_fingerprint;
        also_read = also;
        std::sort(also_read.begin(), also_read.end());
        if (dir.empty()) return fail("no directory");
        for (size_t i = 1; i <= dir.size(); ++i) {
            if (i < dir.size() && dir[i] != '/') continue;
            const std::string part = dir.substr(0, i);
            if (mkdir(part.c_str(), 0700) != 0 && errno != EEXIST) return fail("cannot make " + part);
        }
        errno = 0;
        struct stat st;
        if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return fail(dir + " is not a directory");
        if (st.st_uid != geteuid()) return fail(dir + " belongs to another account");
        if (st.st_mode & 022) return fail(dir + " can be written by other accounts (chmod go-w)");
        if (access(dir.c_str(), R_OK | W_OK | X_OK) != 0) return fail("no right to write in " + dir);
        const std::string lock = dir + "/.lock";
        lock_fd = ::open(lock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lock_fd < 0) return fail("cannot create " + lock);
        if (flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
            ::close(lock_fd);
            lock_fd = -1;
            errno = 0;
            return fail("another server uses " + dir);
        }
        return true;
    }

    // The conversations of this model in the directory, the oldest first. Files that were not finished are
    // removed; those of another model, and those that cannot be read, are counted and left where they are.
    std::vector<entry> scan(size_t & n_other, size_t & n_bad) {
        std::vector<entry> found;
        n_other = n_bad = 0;
        begin();
        DIR * d = opendir(dir.c_str());
        if (!d) { fail("cannot list " + dir); return found; }
        while (const dirent * de = readdir(d)) {
            const std::string name = de->d_name;
            if (name.size() < 8 || name.compare(0, 2, "p-") != 0) continue;
            const std::string path = dir + "/" + name;
            // no other server is here (the lock): a file without its name was given up
            if (ends_with(name, ".ikpc.tmp")) { unlink(path.c_str()); continue; }
            if (!ends_with(name, ".ikpc")) continue;
            entry e;
            const int r = read_entry(path, e);
            // the name of a file says the fingerprint that is in it
            const std::string mine = r == 0 ? "p-" + hex(e.fingerprint) + "-" : std::string();
            if (r == 0 && name.compare(0, mine.size(), mine) == 0) found.push_back(std::move(e));
            else if (r == 0 || r == 1) ++n_other;
            else ++n_bad;
        }
        closedir(d);
        for (const auto & e : found) next_seq = std::max(next_seq, e.seq + 1);
        return by_age(std::move(found));
    }

    // The oldest first. Under one fingerprint the numbers say the order of writing. The files of several
    // fingerprints (`also_read`: each was numbered from 1) are put among each other by their dates, each
    // fingerprint keeping its own order.
    static std::vector<entry> by_age(std::vector<entry> found) {
        std::sort(found.begin(), found.end(), [] (const entry & a, const entry & b) {
            return a.fingerprint != b.fingerprint ? a.fingerprint < b.fingerprint : a.seq < b.seq;
        });
        std::vector<std::pair<size_t, size_t>> runs; // the files of one fingerprint: next one, end
        for (size_t i = 0; i < found.size(); ++i) {
            if (i == 0 || found[i].fingerprint != found[i - 1].fingerprint) runs.emplace_back(i, i);
            runs.back().second = i + 1;
        }
        if (runs.size() < 2) return found;
        std::vector<entry> sorted;
        sorted.reserve(found.size());
        while (sorted.size() < found.size()) {
            size_t best = runs.size();
            for (size_t r = 0; r < runs.size(); ++r) {
                if (runs[r].first == runs[r].second) continue;
                if (best == runs.size()) { best = r; continue; }
                const entry & a = found[runs[r].first];
                const entry & b = found[runs[best].first];
                if (a.written_ns < b.written_ns || (a.written_ns == b.written_ns && a.seq < b.seq)) best = r;
            }
            sorted.push_back(std::move(found[runs[best].first++]));
        }
        return sorted;
    }

    // Writes a conversation. fill is given the place of the state_size bytes of the state, and says whether it
    // put them there. On failure nothing is left in the directory, and no_room tells whether room was missing.
    bool write(const std::string & meta, size_t state_size, const std::function<bool(uint8_t *)> & fill,
               const std::vector<ckpt_in> & ckpts, entry & out) {
        begin();
        const uint64_t page = page_size();
        header h;
        memset(&h, 0, sizeof(h));
        h.magic       = k_magic;
        h.version     = k_version;
        h.fingerprint = fingerprint;
        h.meta_size   = meta.size();
        h.state_off   = (sizeof(h) + meta.size() + page - 1)/page*page; // the state starts on a page
        h.state_size  = state_size;
        h.n_ckpt      = ckpts.size();
        for (const auto & c : ckpts) h.ckpt_size += c.size;
        h.total_size  = h.state_off + state_size + ckpts.size()*sizeof(ckpt_record) + h.ckpt_size + sizeof(uint32_t);

        // a number that no file of the directory has
        std::string path;
        for (;; ++next_seq) {
            path = dir + "/" + prefix() + hex(next_seq) + ".ikpc";
            if (access(path.c_str(), F_OK) != 0) break;
        }
        errno = 0;
        h.seq = next_seq;
        const std::string tmp = path + ".tmp";
        const int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0600);
        if (fd < 0) return fail("cannot create " + tmp);
        bool ok = reserve(fd, h.total_size);
        if (!ok) {
            no_room = errno == ENOSPC || errno == EFBIG || errno == EDQUOT || errno == ENOMEM;
            fail("no room for " + std::to_string(h.total_size >> 20) + " MiB in " + dir);
        }
        ok = ok && put(fd, &h, sizeof(h), 0) && put(fd, meta.data(), meta.size(), sizeof(h));
        if (ok && state_size > 0) ok = put_state(fd, h.state_off, state_size, fill, tmp);
        uint64_t at = h.state_off + state_size;
        for (size_t i = 0; ok && i < ckpts.size(); ++i) {
            ckpt_record r;
            memset(&r, 0, sizeof(r));
            r.pos_min = ckpts[i].pos_min; r.pos_max = ckpts[i].pos_max; r.n_tokens = ckpts[i].n_tokens;
            r.size = ckpts[i].size; r.prompt_end = ckpts[i].prompt_end ? 1 : 0;
            ok = put(fd, &r, sizeof(r), at) && put(fd, ckpts[i].data, ckpts[i].size, at + sizeof(r));
            at += sizeof(r) + ckpts[i].size;
        }
        const uint32_t end = k_end;
        ok = ok && put(fd, &end, sizeof(end), at);
        struct stat st;
        const int64_t written_ns = fstat(fd, &st) == 0 ? ns_of(st) : 0;
        ok = (::close(fd) == 0) && ok;
        // its name, unless another file took it meanwhile: never in the place of a file
        ok = ok && link(tmp.c_str(), path.c_str()) == 0;
        if (!ok) fail("cannot write " + tmp);
        unlink(tmp.c_str());
        if (!ok) return false;
        ++next_seq;
        out = entry();
        out.path = path; out.meta = meta; out.seq = h.seq; out.state_off = h.state_off; out.state_size = state_size;
        out.ckpt_size = h.ckpt_size; out.n_ckpt = h.n_ckpt; out.fingerprint = fingerprint;
        out.written_ns = written_ns;
        return true;
    }

    // The state of a conversation, read where the file is: no copy. nullptr if it cannot be had.
    const uint8_t * map_state(const entry & e) {
        begin();
        const int fd = ::open(e.path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { fail("cannot open " + e.path); return nullptr; }
        void * p = e.state_size > 0 ? mmap(nullptr, e.state_size, PROT_READ, MAP_PRIVATE, fd, (off_t)e.state_off) : MAP_FAILED;
        if (p == MAP_FAILED) fail("cannot map " + e.path);
        ::close(fd);
        return p == MAP_FAILED ? nullptr : (const uint8_t *)p;
    }
    void unmap_state(const entry & e, const uint8_t * p) {
        if (p) munmap((void *)p, e.state_size);
    }

    bool read_checkpoints(const entry & e, std::vector<ckpt_out> & out) {
        out.clear();
        begin();
        const int fd = ::open(e.path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return fail("cannot open " + e.path);
        uint64_t at = e.state_off + e.state_size;
        bool ok = true;
        for (uint64_t i = 0; ok && i < e.n_ckpt; ++i) {
            ckpt_record r;
            ok = get(fd, &r, sizeof(r), at);
            if (!ok) break;
            ckpt_out c;
            c.pos_min = r.pos_min; c.pos_max = r.pos_max; c.n_tokens = r.n_tokens; c.prompt_end = r.prompt_end != 0;
            // a size that the file cannot hold: not a file of ours any more
            if (r.size > e.ckpt_size) { ok = false; break; }
            c.data.resize(r.size);
            ok = get(fd, c.data.data(), r.size, at + sizeof(r));
            at += sizeof(r) + r.size;
            out.push_back(std::move(c));
        }
        if (!ok) fail("cannot read the checkpoints of " + e.path);
        ::close(fd);
        if (!ok) out.clear();
        return ok;
    }

    void remove(const entry & e) {
        if (!e.path.empty()) unlink(e.path.c_str());
    }

private:
    int lock_fd = -1;

    // a call starts without the error of the one before
    void begin() {
        error.clear();
        no_room = false;
        errno = 0;
    }
    // the first reason is the one that is kept
    bool fail(const std::string & why) {
        if (error.empty()) {
            error = why;
            if (errno != 0) { error += ": "; error += strerror(errno); }
        }
        return false;
    }
    static std::string hex(uint64_t v) {
        char buf[20];
        snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
        return buf;
    }
    // the files of this model
    std::string prefix() const {
        return "p-" + hex(fingerprint) + "-";
    }
    static uint64_t page_size() {
        const long p = sysconf(_SC_PAGESIZE);
        return p > 4096 ? (uint64_t)p : 4096;
    }
    static bool ends_with(const std::string & s, const char * suffix) {
        const size_t n = strlen(suffix);
        return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
    }
    // room for the whole file, taken now
    static bool reserve(int fd, uint64_t size) {
#if defined(__linux__)
        int r;
        do { r = posix_fallocate(fd, 0, (off_t)size); } while (r == EINTR);
        errno = r;
        return r == 0;
#else
        return ftruncate(fd, (off_t)size) == 0;
#endif
    }
    // The state, from the caller to the file. Where the room was really taken (Linux) it is written through a
    // mapping, without a copy: writing into a mapped file that has no room left would kill the process, which
    // cannot happen then. Elsewhere it goes through a buffer.
    bool put_state(int fd, uint64_t at, uint64_t size, const std::function<bool(uint8_t *)> & fill, const std::string & name) {
#if defined(__linux__)
        void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)at);
        if (p == MAP_FAILED) return fail("cannot map " + name);
        const bool ok = fill((uint8_t *)p);
        munmap(p, size);
        errno = 0;
        return ok || fail("the state could not be copied");
#else
        std::vector<uint8_t> buf;
        try { buf.resize(size); } catch (const std::exception &) { errno = ENOMEM; return fail("no memory for " + name); }
        if (!fill(buf.data())) { errno = 0; return fail("the state could not be copied"); }
        return put(fd, buf.data(), size, at) || fail("cannot write " + name);
#endif
    }
    static bool put(int fd, const void * data, uint64_t size, uint64_t at) {
        const char * p = (const char *)data;
        while (size > 0) {
            const ssize_t n = pwrite(fd, p, (size_t)std::min<uint64_t>(size, 1u << 30), (off_t)at);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            p += n; at += (uint64_t)n; size -= (uint64_t)n;
        }
        return true;
    }
    static bool get(int fd, void * data, uint64_t size, uint64_t at) {
        char * p = (char *)data;
        while (size > 0) {
            const ssize_t n = pread(fd, p, (size_t)std::min<uint64_t>(size, 1u << 30), (off_t)at);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            p += n; at += (uint64_t)n; size -= (uint64_t)n;
        }
        return true;
    }
    static int64_t ns_of(const struct stat & st) {
#if defined(__APPLE__)
        return (int64_t)st.st_mtimespec.tv_sec*1000000000 + (int64_t)st.st_mtimespec.tv_nsec;
#else
        return (int64_t)st.st_mtim.tv_sec*1000000000 + (int64_t)st.st_mtim.tv_nsec;
#endif
    }
    // 0: a conversation of this model, 1: of another model, 2: not readable, or not a file of this account
    int read_entry(const std::string & path, entry & e) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) return 2;
        struct stat st;
        header h;
        int r = 2;
        if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_uid == geteuid() &&
            get(fd, &h, sizeof(h), 0) && h.magic == k_magic && h.version == k_version &&
            h.total_size == (uint64_t)st.st_size && h.state_off >= sizeof(h) + h.meta_size &&
            h.state_off <= h.total_size && h.state_size <= h.total_size - h.state_off &&
            h.ckpt_size <= h.total_size && h.n_ckpt <= (1u << 20) && h.meta_size < ((uint64_t)1 << 32) &&
            h.state_off%page_size() == 0) {
            uint32_t end = 0;
            if (h.state_off + h.state_size + h.n_ckpt*sizeof(ckpt_record) + h.ckpt_size + sizeof(end) == h.total_size &&
                get(fd, &end, sizeof(end), h.total_size - sizeof(end)) && end == k_end) {
                if (h.fingerprint != fingerprint && !std::binary_search(also_read.begin(), also_read.end(), h.fingerprint)) {
                    r = 1;
                } else {
                    e.meta.resize(h.meta_size);
                    if (get(fd, &e.meta[0], h.meta_size, sizeof(h))) {
                        e.path = path; e.seq = h.seq; e.state_off = h.state_off; e.state_size = h.state_size;
                        e.ckpt_size = h.ckpt_size; e.n_ckpt = h.n_ckpt; e.fingerprint = h.fingerprint;
                        e.written_ns = ns_of(st);
                        r = 0;
                    }
                }
            }
        }
        ::close(fd);
        return r;
    }

#else // not POSIX: the cache stays in the memory of the server

    bool open(const std::string &, uint64_t, const std::vector<uint64_t> & = {}) { error = "not available on this system"; return false; }
    std::vector<entry> scan(size_t & n_other, size_t & n_bad) { n_other = n_bad = 0; return {}; }
    bool write(const std::string &, size_t, const std::function<bool(uint8_t *)> &, const std::vector<ckpt_in> &, entry &) { return false; }
    const uint8_t * map_state(const entry &) { return nullptr; }
    void unmap_state(const entry &, const uint8_t *) {}
    bool read_checkpoints(const entry &, std::vector<ckpt_out> & out) { out.clear(); return false; }
    void remove(const entry &) {}

#endif
};
