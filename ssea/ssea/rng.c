/* file: rng.c */
/* linear congruential random number generator */
#include <stdio.h>
#include <time.h>
#include <math.h>

/* a.k.a. RAND_MAX */
#define MODULUS 2147483647

/* seed generator using time */
int lcg_init_state()
{
	int x;
	x = ((int) time((time_t *) NULL)) % MODULUS;
	return(x);
}

/* integer pseudo-random number based on seed */
int lcg_rand(int seed)
{
	seed = (seed * 1103515245 + 12345) & MODULUS;
	return(seed);
}

double lcg_double(int *seedp)
{
	int seed;
	seed = lcg_rand(*seedp);
	*seedp = seed;
	return(((double) seed) / MODULUS);
}


/*
* copied from numpy.random source code
* log-gamma function to support some of these distributions. The
* algorithm comes from SPECFUN by Shanjie Zhang and Jianming Jin and their
* book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
*/
static double loggam(double x)
{
    double x0, x2, xp, gl, gl0;
    long k, n;

    static double a[10] = {8.333333333333333e-02,-2.777777777777778e-03,
         7.936507936507937e-04,-5.952380952380952e-04,
         8.417508417508418e-04,-1.917526917526918e-03,
         6.410256410256410e-03,-2.955065359477124e-02,
         1.796443723688307e-01,-1.39243221690590e+00};
    x0 = x;
    n = 0;
    if ((x == 1.0) || (x == 2.0))
    {
        return 0.0;
    }
    else if (x <= 7.0)
    {
        n = (long)(7 - x);
        x0 = x + n;
    }
    x2 = 1.0/(x0*x0);
    xp = 2*M_PI;
    gl0 = a[9];
    for (k=8; k>=0; k--)
    {
        gl0 *= x2;
        gl0 += a[k];
    }
    gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0;
    if (x <= 7.0)
    {
        for (k=1; k<=n; k++)
        {
            gl -= log(x0-1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

/* adapted from numpy.random source code */
long lcg_poisson_mult(int *seedp, double lam)
{
	long X;
	double prod, U, enlam;

    enlam = exp(-lam);
    X = 0;
    prod = 1.0;
    while (1)
    {
    	U = lcg_double(seedp);
        prod *= U;
        if (prod > enlam)
        {
            X += 1;
        }
        else
        {
        	/* update seed before returning */
            return X;
        }
    }
}

/* adapted from numpy.random source code */
#define LS2PI 0.91893853320467267
#define TWELFTH 0.083333333333333333333333
long lcg_poisson_ptrs(int *seedp, double lam)
{
    long k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;

    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53*slam;
    a = -0.059 + 0.02483*b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);

    while (1)
    {
    	U = lcg_double(seedp) - 0.5;
    	V = lcg_double(seedp);
        us = 0.5 - fabs(U);
        k = (long)floor((2*a/us + b)*U + lam + 0.43);
        if ((us >= 0.07) && (V <= vr))
        {
            return k;
        }
        if ((k < 0) ||
            ((us < 0.013) && (V > us)))
        {
            continue;
        }
        if ((log(V) + log(invalpha) - log(a/(us*us)+b)) <=
            (-lam + k*loglam - loggam(k+1)))
        {
            return k;
        }
    }
}

/* adapted from numpy.random source code */
long lcg_poisson(int *seedp, double lam)
{
    if (lam >= 10)
    {
        return lcg_poisson_ptrs(seedp, lam);
    }
    else if (lam == 0)
    {
        return 0;
    }
    else
    {
        return lcg_poisson_mult(seedp, lam);
    }
}

double lcg_gauss(int *seedp)
{
	double f, x1, x2, r2;
	do {
		x1 = 2.0*lcg_double(seedp) - 1.0;
		x2 = 2.0*lcg_double(seedp) - 1.0;
		r2 = x1*x1 + x2*x2;
    }
	while (r2 >= 1.0 || r2 == 0.0);
	/* Box-Muller transform */
	f = sqrt(-2.0*log(r2)/r2);
	return f*x2;
}

double lcg_normal(int *seedp, double loc, double scale)
{
    return loc + scale*lcg_gauss(seedp);
}

double lcg_uniform(int *seedp, double loc, double scale)
{
    return loc + scale*lcg_double(seedp);
}

int main()
{
	int i;
	double d;
	int seed;

	seed = lcg_init_state();
	printf("seed is %d", seed);

	for (i = 0; i < 100; i++) {
		seed = lcg_rand(seed);
		d = lcg_poisson(&seed, 30);
		/* d = ((double) seed) / MODULUS; */
		printf("poisson %d %f\n", seed, d);
		d = lcg_normal(&seed, 50, 2);
		printf("normal %d %f\n", seed, d);
	}
 
	return 0;
} 
