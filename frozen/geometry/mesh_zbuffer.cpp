// CPU reference rasterizer: perspective-correct camera-z and near clipping.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

struct V { double x, y, z; };

static int clip_near(const V* in, int n, V* out, double near_z) {
    int count = 0;
    V a = in[n-1]; bool a_in = a.z >= near_z;
    for (int i=0; i<n; ++i) {
        V b=in[i]; bool b_in=b.z >= near_z;
        if (a_in != b_in) {
            double t=(near_z-a.z)/(b.z-a.z);
            out[count++]={a.x+t*(b.x-a.x),a.y+t*(b.y-a.y),near_z};
        }
        if (b_in) out[count++]=b;
        a=b; a_in=b_in;
    }
    return count;
}

static void triangle(V a,V b,V c,const double* k,int w,int h,float* depth) {
    const double ax=k[0]*a.x/a.z+k[2], ay=k[4]*a.y/a.z+k[5];
    const double bx=k[0]*b.x/b.z+k[2], by=k[4]*b.y/b.z+k[5];
    const double cx=k[0]*c.x/c.z+k[2], cy=k[4]*c.y/c.z+k[5];
    double den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy);
    if (!std::isfinite(den) || std::abs(den)<1e-14) return;
    double lo_x=std::min({ax,bx,cx}), hi_x=std::max({ax,bx,cx});
    double lo_y=std::min({ay,by,cy}), hi_y=std::max({ay,by,cy});
    if (hi_x<0 || hi_y<0 || lo_x>w-1 || lo_y>h-1) return;
    int x0=(int)std::max(0.0,std::ceil(lo_x));
    int x1=(int)std::min((double)w-1,std::floor(hi_x));
    int y0=(int)std::max(0.0,std::ceil(lo_y));
    int y1=(int)std::min((double)h-1,std::floor(hi_y));
    for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x) {
        double u=((by-cy)*(x-cx)+(cx-bx)*(y-cy))/den;
        double v=((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/den;
        double t=1-u-v;
        if(u>=-1e-9 && v>=-1e-9 && t>=-1e-9) {
            double iz=u/a.z+v/b.z+t/c.z;
            if(iz>0) {
                float z=(float)(1/iz);
                float& current=depth[(int64_t)y*w+x];
                if(z<current) current=z;
            }
        }
    }
}

extern "C" int raster_depth(const double* vertices,int64_t nv,const int32_t* faces,
                            int64_t nf,const double* k,int w,int h,double near_z,float* depth) {
    if(!vertices || !faces || !k || !depth || w<1 || h<1 || near_z<=0) return -1;
    std::fill(depth,depth+(int64_t)w*h,std::numeric_limits<float>::infinity());
    for(int64_t i=0;i<nf;++i) {
        V input[3]; bool ok=true;
        for(int j=0;j<3;++j) {
            int32_t index=faces[i*3+j];
            if(index<0 || index>=nv) return -2;
            const double* p=vertices+(int64_t)index*3;
            input[j]={p[0],p[1],p[2]};
            ok=ok && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
        }
        if(!ok) continue;
        V polygon[5]; int n=clip_near(input,3,polygon,near_z);
        for(int j=1;j<n-1;++j) triangle(polygon[0],polygon[j],polygon[j+1],k,w,h,depth);
    }
    return 0;
}
