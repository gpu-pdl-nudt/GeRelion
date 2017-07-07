#include "src/timer.h"
#include <cstdio>

wwTimer::wwTimer()
{
        time = 0;
}

void wwTimer::start()
{
        gettimeofday(&start_time, NULL);
}

void wwTimer::stop()
{
        timeval stop_time;
        gettimeofday(&stop_time, NULL);
        time += (stop_time.tv_sec - start_time.tv_sec) + (stop_time.tv_usec - start_time.tv_usec) * 1e-6;
}

timeval start_time[MAX_LINE], stop_time[MAX_LINE];
long int t[MAX_LINE];
long int c[MAX_LINE];

int LAST[10000], LAST_P;