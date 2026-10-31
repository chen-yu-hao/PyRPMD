// PyRPMD MIT. Linux C ABI adapter, persistent isolated native Caracal worker.
// Model data are embedded at build time. No per-step Python or shell calls.
#include "pes_api_v1.h"
#include "model_data.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <poll.h>
#include <set>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
extern char **environ;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static void error_message(pyrpmd_error_v1* e, const char* text) noexcept {
    if (!e) return;
    const size_t length = std::strlen(text);
    e->required_capacity = static_cast<int64_t>(length+1);
    if (e->message && e->capacity > 0) {
        const size_t n = std::min(length, static_cast<size_t>(e->capacity-1));
        std::memcpy(e->message, text, n); e->message[n] = 0;
    }
}
struct pyrpmd_pes_context_v1 {
    int socket = -1;
    pid_t child = -1;
    pid_t owner = getpid();
    std::string directory;
    bool failed = false;
    std::mutex mutex;
    ~pyrpmd_pes_context_v1() {
        // A forked copy must never kill/reap its parent's worker.
        if (socket >= 0) close(socket);
        if (getpid() != owner) return;
        if (child > 0) {
            kill(child, SIGTERM);
            int status;
            bool reaped = false;
            for (int i=0; i<50; ++i) {
                pid_t r = waitpid(child, &status, WNOHANG);
                if (r == child || (r < 0 && errno == ECHILD)) { reaped = true; break; }
                usleep(1000);
            }
            if (!reaped) { kill(child, SIGKILL); while(waitpid(child, &status, 0)<0 && errno==EINTR) {} }
        }
        try {
            std::error_code ec;
            if (!directory.empty()) fs::remove_all(directory, ec);
        } catch (...) {
            // Destruction must not propagate an allocation/filesystem error
            // through the C ABI, including while another exception is active.
        }
    }
};
static std::string log_tail(const pyrpmd_pes_context_v1* c) {
    std::ifstream f(c->directory+"/worker.log", std::ios::binary);
    if (!f) return "";
    f.seekg(0, std::ios::end);
    const std::streamoff length = f.tellg();
    f.seekg(length > 2000 ? length-std::streamoff(2000) : std::streamoff(0));
    return std::string(std::istreambuf_iterator<char>(f), {});
}
static void failure_message(pyrpmd_error_v1* e, const char* message,
                            const pyrpmd_pes_context_v1* c) noexcept {
    try {
        const std::string detail = std::string(message)+(c ? "\n"+log_tail(c) : "");
        error_message(e, detail.c_str());
    } catch (...) {
        // Diagnostics are best effort when allocation or reading the worker
        // log fails. The original error still reaches the caller safely.
        error_message(e, message);
    }
}
static void transfer(pyrpmd_pes_context_v1* c, void* ptr, size_t n, bool sending, Clock::time_point deadline) {
    char* data = static_cast<char*>(ptr);
    while(n) {
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline-Clock::now()).count();
        if (remaining <= 0) throw std::runtime_error("Caracal worker timed out");
        pollfd p{c->socket, static_cast<short>(sending ? POLLOUT : POLLIN), 0};
        int rc = poll(&p, 1, static_cast<int>(std::min<int64_t>(remaining, INT32_MAX)));
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) throw std::runtime_error("Caracal worker timeout/poll failure");
        ssize_t used = sending ? send(c->socket, data, n, MSG_NOSIGNAL | MSG_DONTWAIT)
                               : recv(c->socket, data, n, MSG_DONTWAIT);
        if (used < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (used <= 0) throw std::runtime_error("Caracal worker exited or closed its socket");
        data += used; n -= static_cast<size_t>(used);
    }
}
static std::string compact_json(const char* data, int64_t n) {
    std::string out;
    bool quoted=false, escaped=false;
    for(int64_t i=0; i<n; ++i) {
        char c=data[i];
        if (!quoted && (c==' ' || c=='\n' || c=='\r' || c=='\t')) continue;
        out += c;
        if (!escaped && c=='"') quoted=!quoted;
        if (quoted && !escaped && c=='\\') escaped=true; else escaped=false;
    }
    if (quoted) throw std::runtime_error("unterminated JSON string");
    return out;
}
static void validate_config(const pyrpmd_create_options_v1* o) {
    if (o->config_json_bytes < 0 || o->config_json_bytes > 1048576 ||
        (o->config_json_bytes && !o->config_json)) throw std::runtime_error("invalid JSON buffer");
    std::string s = compact_json(o->config_json, o->config_json_bytes);
    if (s == "{}") return;
    if(s.size()<2 || s.front()!='{' || s.back()!='}') throw std::runtime_error("config must be a JSON object");
    size_t pos=1;
    std::set<std::string> seen;
    while(pos<s.size()-1) {
        bool matched=false;
        for(const auto& item : IDENTITIES) {
            const std::string prefix=std::string("\"")+item.key+"\":";
            const std::string pair=prefix+item.value;
            if(s.compare(pos,pair.size(),pair)==0) {
                if(!seen.insert(item.key).second) throw std::runtime_error("duplicate identity key");
                pos+=pair.size(); matched=true; break;
            }
        }
        if(!matched) throw std::runtime_error("unknown config key or mismatched molecular identity");
        if(pos==s.size()-1) break;
        if(s[pos]!=',' || pos+1==s.size()-1) throw std::runtime_error("malformed config delimiter");
        ++pos;
    }
}
extern "C" int32_t pyrpmd_pes_abi_version_v1(void) { return 1; }
extern "C" pyrpmd_rc_v1 pyrpmd_pes_create_v1(const pyrpmd_create_options_v1* o,
        pyrpmd_pes_context_v1** out, pyrpmd_error_v1* e) {
    if(!out) return PYRPMD_INVALID_ARGUMENT_V1;
    *out=nullptr;
    if(!o || o->abi_version!=1 || o->struct_size!=sizeof(*o)) return PYRPMD_ABI_MISMATCH_V1;
    pyrpmd_pes_context_v1* c=nullptr;
    try {
        validate_config(o);
        c=new pyrpmd_pes_context_v1;
        Dl_info d{};
        if(!dladdr(reinterpret_cast<void*>(&pyrpmd_pes_create_v1), &d) || !d.dli_fname)
            throw std::runtime_error("cannot locate native PES library");
        const auto worker=fs::absolute(fs::path(d.dli_fname)).parent_path()/"caracal_worker.x";
        std::ifstream binary(worker, std::ios::binary);
        uint64_t hash=14695981039346656037ULL;
        uint64_t bytes=0;
        char ch;
        while(binary.get(ch)) { hash^=static_cast<unsigned char>(ch); hash*=1099511628211ULL; ++bytes; }
        if(hash!=WORKER_FNV || bytes!=WORKER_BYTES) throw std::runtime_error("Caracal worker integrity mismatch");
        char temp[]="/tmp/pyrpmd-caracal-XXXXXX";
        char* dirname=mkdtemp(temp);
        if(!dirname) throw std::runtime_error("cannot create isolated work directory");
        c->directory=dirname;
        for(const auto& f : MODEL_FILES) {
            std::ofstream stream(c->directory+"/"+f.name, std::ios::binary);
            stream << f.content;
            if(!stream) throw std::runtime_error("cannot write embedded model input");
        }
        int sv[2];
        if(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv)) throw std::runtime_error("socketpair failed");
        c->socket=sv[0];
        int log=open((c->directory+"/worker.log").c_str(), O_CREAT|O_WRONLY|O_TRUNC|O_CLOEXEC, 0600);
        if(log<0) { close(sv[1]); throw std::runtime_error("cannot open worker log"); }
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addclose(&actions, sv[0]);
        posix_spawn_file_actions_adddup2(&actions, sv[1], 3);
        if(sv[1]!=3) posix_spawn_file_actions_addclose(&actions, sv[1]);
        posix_spawn_file_actions_adddup2(&actions, log, STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&actions, log, STDERR_FILENO);
        if(log>3) posix_spawn_file_actions_addclose(&actions, log);
        posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
        std::string exe=worker.string();
        char key[]="model.key";
        std::string force_mode=FORCE_MODE, force_step=FORCE_STEP;
        char* argv[]={exe.data(), key, c->directory.data(), force_mode.data(), force_step.data(), nullptr};
        int rc=posix_spawn(&c->child, exe.c_str(), &actions, nullptr, argv, environ);
        posix_spawn_file_actions_destroy(&actions);
        close(log); close(sv[1]);
        if(rc) { c->child=-1; throw std::runtime_error("cannot start native Caracal worker"); }
        int64_t n=0;
        transfer(c, &n, sizeof(n), false, Clock::now()+std::chrono::milliseconds(TIMEOUT_MS));
        if(n!=NATOMS) throw std::runtime_error("native worker atom count mismatch");
        *out=c; error_message(e, ""); return PYRPMD_OK_V1;
    } catch(const std::exception& ex) {
        failure_message(e, ex.what(), c);
        delete c; return PYRPMD_BACKEND_FAILURE_V1;
    } catch(...) { delete c; error_message(e,"native creation failed"); return PYRPMD_INTERNAL_ERROR_V1; }
}
extern "C" pyrpmd_rc_v1 pyrpmd_pes_metadata_v1(const pyrpmd_pes_context_v1* c, char* out,
        int64_t capacity, int64_t* required, pyrpmd_error_v1* e) {
    if(!c || !required || capacity<0) return PYRPMD_INVALID_ARGUMENT_V1;
    *required=sizeof(METADATA);
    if(capacity<*required) { if(out && capacity) out[0]=0; return PYRPMD_BUFFER_TOO_SMALL_V1; }
    if(!out) return PYRPMD_INVALID_ARGUMENT_V1;
    std::memcpy(out,METADATA,sizeof(METADATA)); error_message(e, ""); return PYRPMD_OK_V1;
}
extern "C" pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1(pyrpmd_pes_context_v1* c,
        const pyrpmd_pes_input_v1* x, pyrpmd_pes_output_v1* y, pyrpmd_error_v1* e) {
    if(!c || !x || !y) return PYRPMD_INVALID_ARGUMENT_V1;
    if(x->abi_version!=1 || y->abi_version!=1 || x->struct_size!=sizeof(*x) || y->struct_size!=sizeof(*y))
        return PYRPMD_ABI_MISMATCH_V1;
    const int64_t b=x->nstructures, d=3*NATOMS;
    if(b<1 || b>1000000 || b>10000000/NATOMS || x->natoms!=NATOMS || x->layout!=1 || x->scalar_type!=1 ||
       x->positions_count!=b*d || y->energies_count!=b || y->forces_count!=b*d || y->structure_status_count!=b ||
       !x->positions || !y->energies || !y->forces || !y->structure_status) return PYRPMD_INVALID_ARGUMENT_V1;
    if(y->structure_messages) {
        if(y->structure_message_stride<1 || b>INT64_MAX/y->structure_message_stride ||
           y->structure_messages_capacity<b*y->structure_message_stride) return PYRPMD_INVALID_ARGUMENT_V1;
    } else if(y->structure_message_stride || y->structure_messages_capacity) return PYRPMD_INVALID_ARGUMENT_V1;
    // Keep the lock alive through exception handling. A failed transfer must
    // poison the context before another evaluator can start a transaction.
    std::unique_lock<std::mutex> lock(c->mutex, std::defer_lock);
    try {
        if(c->owner!=getpid()) throw std::runtime_error("PES context cannot be inherited across fork; recreate it");
        lock.lock();
        if(c->failed) throw std::runtime_error("Caracal worker is unusable after a failed transaction");
        std::fill(y->energies,y->energies+b,std::numeric_limits<double>::quiet_NaN());
        std::fill(y->forces,y->forces+b*d,std::numeric_limits<double>::quiet_NaN());
        std::vector<int64_t> indices;
        std::vector<double> q;
        for(int64_t i=0;i<b;++i) {
            bool finite=true, domain=true;
            for(int64_t j=0;j<d;++j) finite=finite && std::isfinite(x->positions[i*d+j]);
            if(finite) for(int64_t a=0;a<NATOMS;++a) for(int64_t z=0;z<a;++z) {
                double r2=0; for(int k=0;k<3;++k) { double v=x->positions[i*d+3*a+k]-x->positions[i*d+3*z+k]; r2+=v*v; }
                domain=domain && r2>1e-12;
            }
            y->structure_status[i]=!finite ? PYRPMD_STRUCTURE_NONFINITE_V1 :
                                   !domain ? PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V1 : PYRPMD_STRUCTURE_OK_V1;
            if(y->structure_messages) y->structure_messages[i*y->structure_message_stride]=0;
            if(finite && domain) { indices.push_back(i); q.insert(q.end(),x->positions+i*d,x->positions+(i+1)*d); }
        }
        int64_t count=static_cast<int64_t>(indices.size());
        if(count) {
            std::vector<double> energies(count), forces(count*d);
            auto deadline=Clock::now()+std::chrono::milliseconds(TIMEOUT_MS);
            transfer(c,&count,sizeof(count),true,deadline);
            transfer(c,q.data(),q.size()*sizeof(double),true,deadline);
            transfer(c,energies.data(),energies.size()*sizeof(double),false,deadline);
            transfer(c,forces.data(),forces.size()*sizeof(double),false,deadline);
            for(int64_t j=0;j<count;++j) {
                int64_t i=indices[j]; bool finite=std::isfinite(energies[j]);
                for(int64_t k=0;k<d;++k) finite=finite && std::isfinite(forces[j*d+k]);
                if(!finite) { y->structure_status[i]=PYRPMD_STRUCTURE_NONFINITE_V1; continue; }
                y->energies[i]=energies[j]; std::copy(forces.data()+j*d,forces.data()+(j+1)*d,y->forces+i*d);
            }
        }
        error_message(e, ""); return PYRPMD_OK_V1;
    } catch(const std::exception& ex) {
        c->failed=true; failure_message(e,ex.what(),c); return PYRPMD_BACKEND_FAILURE_V1;
    } catch(...) { c->failed=true; error_message(e,"native evaluation failed"); return PYRPMD_INTERNAL_ERROR_V1; }
}
extern "C" void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1* c) { delete c; }
