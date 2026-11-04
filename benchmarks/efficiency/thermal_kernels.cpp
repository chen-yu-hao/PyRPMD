// Compile with the same -O3 / OpenMP / FMA flags as the Python extension.
#include <cstddef>
using Py_ssize_t=std::ptrdiff_t;
#include "thermal.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

namespace thermal {
struct KernelProfile {
    template<class F> static double measure(F fn,size_t repeats){
        fn();
        const auto start=std::chrono::steady_clock::now();
        for(size_t i=0;i<repeats;++i)fn();
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
    }
    static void run(Engine& e,size_t repeats){
        std::cout<<std::setprecision(12)<<"{";
        bool first=true;
        auto emit=[&](const char* name,double value){if(!first)std::cout<<",";first=false;std::cout<<"\""<<name<<"\":"<<value;};
        emit("four_transforms_seconds",measure([&]{e.to_modes(e.s.q,e.qn);e.to_modes(e.s.p,e.pn);e.from_modes(e.qn,e.s.q);e.from_modes(e.pn,e.s.p);},repeats));
        emit("pes_seconds",measure([&]{e.refresh();},repeats));
        emit("two_kicks_seconds",measure([&]{e.kick(.5*e.c.dt);e.kick(.5*e.c.dt);},repeats));
        Vec noise(e.l);
        emit("noise_generation_seconds",measure([&]{const auto stream=e.begin_random();
            if(e.c.philox)e.parallel_for(e.c.beads*((e.d+1)/2),[&](size_t i){
                const size_t k=i/((e.d+1)/2),pair=i%((e.d+1)/2);
                const auto values=stream.normal_pair(k,pair,10);noise[k*e.d+2*pair]=values[0];
                if(2*pair+1<e.d)noise[k*e.d+2*pair+1]=values[1];
            });
            else for(size_t i=0;i<e.l;++i)noise[i]=e.g();},repeats));
        emit("noise_application_seconds",measure([&]{e.parallel_for(e.l,[&](size_t i){e.pn[i]=.9*e.pn[i]+.1*noise[i];});},repeats));
        emit("bath_total_seconds",measure([&]{e.bath(e.c.dt,true);},repeats));
        Vec row(ncolumns),centroid(e.d);
        emit("observations_seconds",measure([&]{e.observe(row.data(),centroid.data());},repeats));
        State backup;
        emit("rollback_copy_seconds",measure([&]{backup=e.s;},repeats));
        emit("middle_map_seconds",measure([&]{e.free_bath_free();},repeats));
        emit("step_seconds",measure([&]{e.one_step();},repeats));
        // STREAM-style triad, 192 MiB working set; report 24 bytes/element.
        const size_t size=8*1024*1024;
        Vec a(size,0.),b(size,1.),c(size,2.);
        const double stream=measure([&]{e.parallel_for(size,[&](size_t i){a[i]=b[i]+3.*c[i];});},repeats);
        emit("stream_triad_GB_per_second",24.*size/stream/1e9);
        volatile double retained=a[size/2]+backup.p[0]+centroid[0]+noise[0];(void)retained;
        std::cout<<"}\n";
    }
};
}
int main(int argc,char** argv){
    thermal::Config config;
    config.atoms=argc>1?std::stoull(argv[1]):4096;
    config.beads=argc>2?std::stoull(argv[2]):32;
    const std::string rng=argc>3?argv[3]:"mt19937_64";
    if((rng!="mt19937_64"&&rng!="philox-v1")||config.atoms==0||config.beads==0){
        std::cerr<<"positive atoms/beads and mt19937_64 or philox-v1 are required\n";return 2;
    }
    config.philox=rng=="philox-v1";config.seed=1749;
    config.masses.assign(config.atoms,2.);config.fixed.assign(config.atoms,0);
    config.beta=2.;config.dt=.01;config.potential=1;config.parameters={.5};
    thermal::Vec q(config.atoms*config.beads*3,.1),p(q.size(),0.);
    thermal::Engine engine(config,q,p,thermal::identity());
    thermal::KernelProfile::run(engine,20);
}
