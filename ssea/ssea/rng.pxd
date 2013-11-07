# file: rng.pxd
cdef extern from "rng.h":
    int lcg_init_state()
    int lcg_rand(int seed)
    double lcg_double(int *seedp)
    long lcg_poisson_mult(int *seedp, double lam)
    long lcg_poisson_ptrs(int *seedp, double lam)
    long lcg_poisson(int *seedp, double lam)
    double lcg_gauss(int *seedp)
    double lcg_normal(int *seedp, double loc, double scale)
    double lcg_uniform(int *seedp, double loc, double scale)
