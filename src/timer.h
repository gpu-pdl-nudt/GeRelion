#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

#define TIMER_ENABLE

class wwTimer;

class wwTimer {

public:
        timeval start_time;
        double time_usec;

public:
        wwTimer();
        void start();
        void stop();

        double time;
};

#define MAX_LINE 70000

extern timeval start_time[MAX_LINE], stop_time[MAX_LINE];
extern long int t[MAX_LINE];
extern long int c[MAX_LINE];

extern int LAST[10000], LAST_P;


#ifdef TIMER_ENABLE
#define TIMER_START \
gettimeofday(&start_time[__LINE__], NULL); \
c[__LINE__]++; \
LAST[++LAST_P] = __LINE__;
#else
#define TIMER_START
#endif

#ifdef TIMER_ENABLE
#define TIMER_END \
gettimeofday(&stop_time[LAST[LAST_P]], NULL); \
t[LAST[LAST_P]] += (stop_time[LAST[LAST_P]].tv_sec - start_time[LAST[LAST_P]].tv_sec)*1000000 + (stop_time[LAST[LAST_P]].tv_usec - start_time[LAST[LAST_P]].tv_usec); \
LAST_P--;
#else
#define TIMER_END
#endif

#ifdef TIMER_ENABLE
#define TIMER_COUNT \
for (int i = 0; i < MAX_LINE; i++) if (c[i] > 0) printf("%d:%d ", i, c[i]); printf("\n");
#else
#define TIMER_COUNT
#endif

#ifdef TIMER_ENABLE	
#define TIMER_TIME \
for (int i = 0; i < MAX_LINE; i++) if (c[i] > 0) printf("%d:%lf ", i, t[i]/1000000.); printf("\n");
#else
#define TIMER_TIME
#endif

#define TIMER_CLEAR \
memset(c, 0, sizeof(long int)*MAX_LINE); \
memset(t, 0, sizeof(long int)*MAX_LINE);

#endif
