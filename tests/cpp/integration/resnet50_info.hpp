#pragma once
#include <vector>

std::vector<int> resnet50_grad_sizes()
{
    return {
        1000,    2048000, 2048, 2048, 2048,    1048576, 512,  512,
        512,     2359296, 512,  512,  512,     1048576, 2048, 2048,
        2048,    1048576, 512,  512,  512,     2359296, 512,  512,
        512,     1048576, 2048, 2048, 2048,    2048,    2048, 2048,
        1048576, 512,     512,  512,  2097152, 2359296, 512,  512,
        512,     524288,  1024, 1024, 1024,    262144,  256,  256,
        256,     589824,  256,  256,  256,     262144,  1024, 1024,
        1024,    262144,  256,  256,  256,     589824,  256,  256,
        256,     262144,  1024, 1024, 1024,    262144,  256,  256,
        256,     589824,  256,  256,  256,     262144,  1024, 1024,
        1024,    262144,  256,  256,  256,     589824,  256,  256,
        256,     262144,  1024, 1024, 1024,    262144,  256,  256,
        256,     589824,  256,  256,  256,     262144,  1024, 1024,
        1024,    1024,    1024, 1024, 262144,  524288,  256,  256,
        256,     589824,  256,  256,  256,     131072,  512,  512,
        512,     65536,   128,  128,  128,     147456,  128,  128,
        128,     65536,   512,  512,  512,     65536,   128,  128,
        128,     147456,  128,  128,  128,     65536,   512,  512,
        512,     65536,   128,  128,  128,     147456,  128,  128,
        128,     65536,   512,  512,  512,     512,     512,  512,
        65536,   131072,  128,  128,  128,     147456,  128,  128,
        128,     32768,   256,  256,  256,     16384,   64,   64,
        64,      36864,   64,   64,   64,      16384,   256,  256,
        256,     16384,   64,   64,   64,      36864,   64,   64,
        64,      16384,   256,  256,  256,     256,     256,  256,
        16384,   16384,   64,   64,   64,      36864,   64,   64,
        64,      4096,    64,   64,   64,      9408,
    };
}