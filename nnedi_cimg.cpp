#define cimg_display 0
#include <CImg.h>
#include <pthread.h>
#include <unistd.h>
#include "nnedi.h"
#include "tables.h"
using namespace cimg_library;

typedef struct {
    nnedi_t *ctx;
    CImg<uint8_t> *dst, *src;
    int plane;
} arg_t;

static void worker(arg_t *a)
{
    nnedi_upscale_2x(a->ctx, a->dst->data(0,0,0,a->plane), a->src->data(0,0,0,a->plane), a->src->_width, a->src->_height, a->dst->_width, a->src->_width);
}

CImg<uint8_t> nnedi_upscale_2x_cimg(CImg<uint8_t> src, int nns, int threads)
{
    if(src._depth != 1)
        throw "";
    if(threads <= 0 )
        threads = sysconf(_SC_NPROCESSORS_ONLN);
    nnedi_t *ctx;
    CImg<uint8_t> dst(src._width*2, src._height*2, 1, src._spectrum);
    if(threads < src._spectrum || src._spectrum == 1) {
        ctx = nnedi_config(nns, threads);
        cimg_forC(src, i) {
            arg_t a = {ctx, &dst, &src, i};
            worker(&a);
        }
    } else {
        ctx = nnedi_config(nns, (threads+1)/src._spectrum);
        pthread_t handle[src._spectrum];
        arg_t arg[src._spectrum];
        for(int i=0; i<src._spectrum; i++) {
            arg[i] = (arg_t){ctx, &dst, &src, i};
            pthread_create(&handle[i], NULL, (void*(*)(void*))worker, &arg[i]);
        }
        for(int i=0; i<src._spectrum; i++)
            pthread_join(handle[i], NULL);
    }
    free(ctx);
    return dst;
}
