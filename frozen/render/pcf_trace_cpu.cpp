// Exact-parameter CPU implementation of depth_fusion.trace_lattice.
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>

extern "C" void trace_pcf(const double* phase,const double* directions,int64_t count,
                           double spacing,double far_cells,int steps,int threads,
                           double* depth,double* terminal_t,uint8_t* unresolved) {
    #pragma omp parallel for num_threads(threads) schedule(dynamic,512)
    for(int64_t i=0;i<count;++i) {
        double t=.002;bool hit=false;bool active=true;
        const double *ray=directions+i*3;
        for(int k=0;k<steps;++k) {
            double x=phase[0]+t*ray[0],y=phase[1]+t*ray[1],z=phase[2]+t*ray[2];
            x-=std::floor(x+.5);y-=std::floor(y+.5);z-=std::floor(z+.5);
            double sx=x*x,sy=y*y,sz=z*z;
            double sphere=std::sqrt(sx+sy+sz)-.05;
            double rod=std::sqrt(std::min({sx+sy,sx+sz,sy+sz}))-.008;
            double d=std::min(sphere,rod);
            if(std::abs(d)<.00015) { hit=true;active=false;break; }
            t+=std::max(std::abs(d)*.95,.000075);
            if(t>=far_cells) {active=false;break;}
        }
        terminal_t[i]=t;unresolved[i]=(uint8_t)active;
        depth[i]=hit?t*spacing:std::numeric_limits<double>::infinity();
    }
}
