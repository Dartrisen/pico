#pragma once

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>

namespace bench
{

    /**
     * @brief Read hardware performance counter via /proc/self/stat
     *
     * This is a portable fallback when perf is not available.
     * Measures page faults, which correlate with memory bandwidth usage.
     */
    class PerfStats
    {
    public:
        struct Snapshot
        {
            uint64_t minflt; // Minor page faults
            uint64_t majflt; // Major page faults
            uint64_t vsize;  // Virtual memory size
            uint64_t rss;    // Resident set size (pages)
        };

        static Snapshot read_stat()
        {
            Snapshot s;
            FILE*    f = fopen("/proc/self/stat", "r");
            if (!f)
                return {0, 0, 0, 0};

            // Parse: pid comm state ppid pgrp session tty_nr tpgid flags
            //        minflt cminflt majflt cmajflt utime stime cutime cstime
            //        priority nice num_threads itrealvalue starttime vsize rss ...
            uint64_t pid;
            char     comm[256];
            char     state;
            int      ppid, pgrp, session, tty, tpgid, flags;
            uint64_t minflt, cminflt, majflt, cmajflt;
            uint64_t utime, stime, cutime, cstime;
            int      priority, nice, num_threads;
            uint64_t itrealvalue, starttime, vsize, rss;

            int ret = fscanf(f, "%lu %s %c %d %d %d %d %d %d %lu %lu %lu %lu %lu %lu %lu %lu %d %d %d %lu %lu %lu %lu",
                             &pid, comm, &state, &ppid, &pgrp, &session, &tty, &tpgid, &flags, &minflt, &cminflt,
                             &majflt, &cmajflt, &utime, &stime, &cutime, &cstime, &priority, &nice, &num_threads,
                             &itrealvalue, &starttime, &vsize, &rss);
            fclose(f);

            if (ret < 24)
                return {0, 0, 0, 0};

            s.minflt = minflt;
            s.majflt = majflt;
            s.vsize  = vsize;
            s.rss    = rss;
            return s;
        }

        static void print_delta(const Snapshot& before, const Snapshot& after)
        {
            uint64_t minflt_delta = (after.minflt > before.minflt) ? (after.minflt - before.minflt) : 0;
            uint64_t majflt_delta = (after.majflt > before.majflt) ? (after.majflt - before.majflt) : 0;

            std::cout << "  Minor faults: " << minflt_delta << "\n";
            std::cout << "  Major faults: " << majflt_delta << "\n";
            std::cout << "  Memory delta (KB): " << ((after.rss - before.rss) * 4) << "\n";
        }
    };

    /**
     * @brief Estimate memory bandwidth from data size and time
     */
    class BandwidthCounter
    {
    public:
        BandwidthCounter(uint64_t bytes_accessed, double time_sec)
                : bytes_accessed_(bytes_accessed), time_sec_(time_sec)
        {
        }

        double bandwidth_gbps() const
        {
            return (bytes_accessed_ * 1e-9) / time_sec_;
        }

        double bandwidth_mbs() const
        {
            return (bytes_accessed_ * 1e-6) / time_sec_;
        }

        void print() const
        {
            std::cout << std::fixed << std::setprecision(2) << "  Estimated BW: " << bandwidth_gbps() << " GB/s\n";
        }

    private:
        uint64_t bytes_accessed_;
        double   time_sec_;
    };

} // namespace bench
