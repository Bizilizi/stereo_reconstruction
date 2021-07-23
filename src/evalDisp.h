
#ifndef STEREO_RECONSTRUCTION_EVALDISP_H
#define STEREO_RECONSTRUCTION_EVALDISP_H

#include "opencv2/core.hpp"
#include <cassert>
#include <math.h>
#include <stdlib.h>

/*
 * Note: Adopted from imageLib in the Middlebury SDK
 * */

void evaldisp(cv::Mat disp, cv::Mat gtdisp, cv::Mat mask, float badthresh, float maxdisp, int rounddisp)
{
    cv::Size gtShape = gtdisp.size();
    cv::Size sh = disp.size();
    cv::Size maskShape = mask.size();
    assert (gtShape == sh);
    assert (gtShape == maskShape);

    int n = 0;
    int bad = 0;
    int invalid = 0;
    float serr = 0;
    for (int y = 0; y < gtShape.height; y++) {
        for (int x = 0; x < gtShape.width; x++) {
            float gt = gtdisp.at<float>(y, x);
            if (gt == INFINITY)                      // unknown
                continue;
            float d = disp.at<float>(y, x);
            bool valid = (d != INFINITY);
            if (valid)
                d = std::max(0.0f, std::min(maxdisp, d));
            if (valid && rounddisp)
                d = round(d);
            float err = std::abs(d - gt);
            if (mask.at<uint8_t>(y, x) != 255) {
                // do not evaluate
            } else {
                n++;
                if (valid) {
                    serr += err;
                    if (err > badthresh)
                        bad++;
                } else {
                    invalid++;
                }
            }
        }
    }

    float badpercent =  100.0 * bad / n;
    float invalidpercent =  100.0 * invalid / n;
    float totalbadpercent =  100.0 * ( bad + invalid ) / n;
    float avgErr = serr / (n - invalid);
    std::cout << "number of evaluated: " << n << "\n";
    printf("%4.1f  %6.2f  %6.2f   %6.2f  %6.2f\n",   100.0*n/(gtShape.width * gtShape.height),
           badpercent, invalidpercent, totalbadpercent, avgErr);
}

#endif //STEREO_RECONSTRUCTION_EVALDISP_H
