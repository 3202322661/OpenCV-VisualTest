#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::Mat::zeros(200, 300, CV_8UC3);
    cv::putText(img, "OpenCV OK", cv::Point(50, 100),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Test", img);
    cv::waitKey(0);
    return 0;
}