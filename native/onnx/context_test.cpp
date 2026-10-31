// Native/OpenMP context isolation test, not a replacement RPMD scheduler.
// SPDX-License-Identifier: MIT
#include "pyrpmd/pes_api_v1.h"
#include <omp.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Context = std::unique_ptr<pyrpmd_pes_context_v1, decltype(&pyrpmd_pes_destroy_v1)>;
struct Result { std::vector<double> e, f; };
Context create(const std::string& path) {
    const auto config = nlohmann::json{{"model",path},{"device","cpu"},{"intra_op_threads",1}}.dump();
    char msg[1024]{};
    pyrpmd_error_v1 error{msg,1024,0};
    pyrpmd_create_options_v1 opts{1,sizeof(opts),config.data(),static_cast<int64_t>(config.size())};
    pyrpmd_pes_context_v1* p = nullptr;
    if (pyrpmd_pes_create_v1(&opts,&p,&error)) throw std::runtime_error(msg);
    return Context(p, pyrpmd_pes_destroy_v1);
}
Result evaluate(pyrpmd_pes_context_v1* p, int64_t batch, double shift) {
    // CI fixture has three atoms. Distinct coordinates per trajectory/batch.
    std::vector<double> q(static_cast<size_t>(batch*9));
    for (size_t j=0;j<q.size();++j) q[j] = shift+0.003*static_cast<double>(j);
    Result r{std::vector<double>(batch),std::vector<double>(q.size())};
    std::vector<int32_t> status(batch);
    pyrpmd_pes_input_v1 in{1,sizeof(in),1,1,batch,3,q.data(),static_cast<int64_t>(q.size())};
    pyrpmd_pes_output_v1 out{1,sizeof(out),r.e.data(),batch,r.f.data(),static_cast<int64_t>(q.size()),status.data(),batch,nullptr,0,0};
    char msg[1024]{}; pyrpmd_error_v1 error{msg,1024,0};
    if (pyrpmd_pes_evaluate_v1(p,&in,&out,&error)) throw std::runtime_error(msg);
    for (const auto s:status) if(s) throw std::runtime_error("frame failure");
    return r;
}
bool equal(const Result& a,const Result& b) {
    if(a.e.size()!=b.e.size() || a.f.size()!=b.f.size()) return false;
    for(size_t i=0;i<a.e.size();++i) if(!std::isfinite(a.e[i]) || std::abs(a.e[i]-b.e[i])>1e-12) return false;
    for(size_t i=0;i<a.f.size();++i) if(!std::isfinite(a.f[i]) || std::abs(a.f[i]-b.f[i])>1e-12) return false;
    return true;
}
int main(int argc,char** argv) {
    if(argc!=3) {std::cerr<<"usage: onnx_context_test model1.onnx model2.onnx\n";return 2;}
    try {
        constexpr int jobs=8;
        std::vector<Context> contexts;
        std::vector<Result> expected;
        for(int i=0;i<jobs;++i) {
            contexts.push_back(create(argv[1+i%2]));
            expected.push_back(evaluate(contexts.back().get(),1+i,0.2+i));
        }
        int failures=0;
        #pragma omp parallel for schedule(dynamic,1) num_threads(4) reduction(+:failures)
        for(int i=0;i<jobs;++i) {
            try {
                for(int step=0;step<50;++step) {
                    // Force a buffer-size change, then check the original batch.
                    evaluate(contexts[i].get(),32,0.1+step);
                    if(!equal(expected[i],evaluate(contexts[i].get(),1+i,0.2+i))) ++failures;
                }
            } catch (...) { ++failures; }
        }
        // Same-context calls are serialized by the plugin mutex, not racy.
        #pragma omp parallel for num_threads(4) reduction(+:failures)
        for(int i=0;i<16;++i) {
            try {if(!equal(expected[0],evaluate(contexts[0].get(),1,0.2))) ++failures;}
            catch (...) {++failures;}
        }
        if(failures) {std::cerr<<failures<<" context failures\n";return 1;}
        std::cout<<"OpenMP: 8 independent contexts, 2 models, 4 threads; isolation and same-context locking passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
