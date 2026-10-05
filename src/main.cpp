// 蓝色装甲板灯条检测与框选
// 流程：视频读取 → 颜色分割 → 掩膜优化 → 轮廓提取 → 几何筛选 → 框选显示
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* kDefaultVideo = "test_video2.webm";
const cv::Scalar kWhite(255, 255, 255);
const cv::Scalar kGreen(0, 255, 0);
const cv::Scalar kAmber(0, 180, 255);

// ---------------------------------------------------------------------------
// 可调参数：全部是“颜色区间 / 面积比例 / 形状比例”，
// 不依赖任何固定像素坐标，也不依赖固定帧号。
// ---------------------------------------------------------------------------
struct DetectParams {
    // 颜色分割：HSV 蓝色区间。实测蓝灯条 H≈101~103，暖色场地灯 H≈7~10 被排除
    int hMin = 90;
    int hMax = 125;
    int sMin = 80;
    int vMin = 60;

    // 掩膜优化：3x3 开运算（仅去孤立噪点，不使用闭运算，理由见参数对比说明）
    int openSize = 3;

    // 轮廓面积占整幅图像的比例，随分辨率自适应
    double minAreaRatio = 2e-5;
    double maxAreaRatio = 0.02;

    // 灯条几何：基于 minAreaRect 的长边/短边
    double minAspect = 1.8;
    double maxAspect = 16.0;
    double minThickness = 2.5;
    double minLength = 10.0;

    // 紧贴度：旋转矩形内属于蓝色掩膜的像素占比，排除不规则或粘连的区域
    double minPurity = 0.45;

    // 面积下限的绝对值兜底（像素），用于剔除画面边缘的细小杂点
    double minAreaPixels = 30.0;
};

struct Bar {
    cv::RotatedRect rect;
    double purity = 0.0;
};

struct FrameResult {
    std::vector<Bar> bars;
    std::vector<std::vector<cv::Point>> contours;   // 全部外层轮廓（演示用）
    std::vector<std::vector<cv::Point>> polygons;   // 多边形近似（演示用）
    cv::Mat mask;
    int maskPixels = 0;
};

bool openVideo(const std::string& path, cv::VideoCapture& cap) {
    if (!cap.open(path, cv::CAP_FFMPEG)) {
        cap.open(path);  // 退回自动选择后端
    }
    return cap.isOpened();
}

// 依次尝试：用户传入的路径、当前目录、可执行文件所在目录的上一级
std::string resolveVideoPath(const std::string& requested, const char* argv0) {
    std::vector<std::string> candidates = {requested};
    if (argv0 != nullptr) {
        const std::string exe(argv0);
        const auto slash = exe.find_last_of('/');
        if (slash != std::string::npos) {
            candidates.push_back(exe.substr(0, slash + 1) + "../" + requested);
        }
    }
    for (const auto& c : candidates) {
        if (std::FILE* f = std::fopen(c.c_str(), "rb")) {
            std::fclose(f);
            return c;
        }
    }
    return requested;
}

double sanitizeFps(double fps) {
    return (std::isfinite(fps) && fps > 0.0) ? fps : 30.0;
}

// ---------------------------------------------------------------------------
// 颜色分割 + 掩膜优化 + 轮廓提取 + 几何筛选
// ---------------------------------------------------------------------------
FrameResult detectBars(const cv::Mat& bgr, const DetectParams& p) {
    FrameResult r;

    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(p.hMin, p.sMin, p.vMin),
                cv::Scalar(p.hMax, 255, 255), r.mask);

    if (p.openSize > 1) {
        cv::morphologyEx(r.mask, r.mask, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_RECT, {p.openSize, p.openSize}));
    }
    r.maskPixels = cv::countNonZero(r.mask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(r.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double imgArea = static_cast<double>(bgr.rows) * bgr.cols;
    const double minArea = std::max(p.minAreaPixels, p.minAreaRatio * imgArea);
    const double maxArea = p.maxAreaRatio * imgArea;

    for (const auto& c : contours) {
        r.contours.push_back(c);

        std::vector<cv::Point> poly;
        cv::approxPolyDP(c, poly, 0.02 * cv::arcLength(c, true), true);
        r.polygons.push_back(poly);

        const double area = cv::contourArea(c);
        if (area < minArea || area > maxArea) {
            continue;
        }

        const cv::RotatedRect rr = cv::minAreaRect(c);
        const double longSide = std::max(rr.size.width, rr.size.height);
        const double shortSide = std::min(rr.size.width, rr.size.height);
        if (shortSide < p.minThickness || longSide < p.minLength) {
            continue;
        }
        const double aspect = longSide / std::max(1.0, shortSide);
        if (aspect < p.minAspect || aspect > p.maxAspect) {
            continue;
        }

        cv::Mat boxMask = cv::Mat::zeros(r.mask.size(), CV_8U);
        cv::Point2f quad[4];
        rr.points(quad);
        std::vector<cv::Point> quadPts;
        quadPts.reserve(4);
        for (const auto& q : quad) {
            quadPts.emplace_back(cvRound(q.x), cvRound(q.y));
        }
        cv::fillConvexPoly(boxMask, quadPts, 255);

        const int inside = cv::countNonZero(boxMask);
        const double purity =
            inside > 0 ? cv::countNonZero(r.mask & boxMask) / static_cast<double>(inside) : 0.0;
        if (purity < p.minPurity) {
            continue;
        }

        r.bars.push_back({rr, purity});
    }
    return r;
}

void drawBars(cv::Mat& img, const FrameResult& r) {
    for (const auto& b : r.bars) {
        cv::Point2f quad[4];
        b.rect.points(quad);
        std::vector<cv::Point> poly;
        poly.reserve(4);
        for (const auto& q : quad) {
            poly.emplace_back(cvRound(q.x), cvRound(q.y));
        }
        cv::polylines(img, std::vector<std::vector<cv::Point>>{poly}, true, kGreen, 2, cv::LINE_AA);
        cv::circle(img, b.rect.center, 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }
}

// 在画面上标注帧号、灯条数量和处理耗时
void drawHud(cv::Mat& img, int frameIndex, int totalFrames, int barCount, double ms) {
    const cv::Rect panel(10, 10, 340, 112);
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, panel, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.55, img, 0.45, 0.0, img);

    char l1[128], l2[128], l3[128];
    std::snprintf(l1, sizeof(l1), "frame : %d / %d", frameIndex, totalFrames);
    std::snprintf(l2, sizeof(l2), "bars  : %d", barCount);
    std::snprintf(l3, sizeof(l3), "detect: %.2f ms", ms);
    cv::putText(img, l1, cv::Point(22, 44), cv::FONT_HERSHEY_SIMPLEX, 0.72, kWhite, 2, cv::LINE_AA);
    cv::putText(img, l2, cv::Point(22, 76), cv::FONT_HERSHEY_SIMPLEX, 0.72,
                barCount > 0 ? kGreen : kAmber, 2, cv::LINE_AA);
    cv::putText(img, l3, cv::Point(22, 108), cv::FONT_HERSHEY_SIMPLEX, 0.72, kWhite, 2, cv::LINE_AA);

    if (barCount == 0) {
        const char* msg = "no valid light bar in this frame";
        int baseline = 0;
        const cv::Size sz = cv::getTextSize(msg, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
        cv::putText(img, msg, cv::Point((img.cols - sz.width) / 2, img.rows - 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, kAmber, 2, cv::LINE_AA);
    }
}

cv::Mat makeCell(const cv::Mat& src, const std::string& label, int w, int h) {
    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(22, 22, 22));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    resized.copyTo(canvas);
    cv::rectangle(canvas, cv::Rect(0, 0, w, 26), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(canvas, label, cv::Point(8, 19), cv::FONT_HERSHEY_SIMPLEX, 0.58, kWhite, 1,
                cv::LINE_AA);
    return canvas;
}

cv::Mat grayCell(const cv::Mat& gray, const std::string& label, int w, int h) {
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    return makeCell(bgr, label, w, h);
}

// 管线演示图：原图 + B/G/R 分通道 + 灰度 + 掩膜 + 轮廓/多边形 + 最终框选
cv::Mat buildPipelineFigure(const cv::Mat& bgr, const FrameResult& r, int frameIndex, double ms) {
    const int cellW = 480, cellH = 308, gap = 6, header = 34;

    std::vector<cv::Mat> cells;
    cv::Mat shown = bgr.clone();
    drawBars(shown, r);
    cells.push_back(makeCell(shown, "1 original + detection", cellW, cellH));

    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);  // ch[0]=B, ch[1]=G, ch[2]=R
    cells.push_back(grayCell(ch[0], "2 Blue channel", cellW, cellH));
    cells.push_back(grayCell(ch[1], "3 Green channel", cellW, cellH));
    cells.push_back(grayCell(ch[2], "4 Red channel", cellW, cellH));

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cells.push_back(grayCell(gray, "5 grayscale", cellW, cellH));
    cells.push_back(grayCell(r.mask, "6 blue mask: HSV inRange + 3x3 open", cellW, cellH));

    cv::Mat contourVis = cv::Mat::zeros(bgr.size(), CV_8UC3);
    cv::drawContours(contourVis, r.contours, -1, cv::Scalar(110, 110, 110), 1, cv::LINE_AA);
    cv::drawContours(contourVis, r.polygons, -1, cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
    cells.push_back(makeCell(contourVis, "7 contours + polygon approx", cellW, cellH));

    cv::Mat boxVis = cv::Mat::zeros(bgr.size(), CV_8UC3);
    drawBars(boxVis, r);
    cells.push_back(makeCell(boxVis, "8 rotated-rect fit (final)", cellW, cellH));

    const int cols = 4, rows = 2;
    cv::Mat fig(header + gap + rows * cellH + (rows - 1) * gap + gap,
                gap + cols * cellW + (cols - 1) * gap + gap, CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t i = 0; i < cells.size(); ++i) {
        const int row = static_cast<int>(i) / cols;
        const int col = static_cast<int>(i) % cols;
        cv::Rect dst(gap + col * (cellW + gap), header + gap + row * (cellH + gap), cellW, cellH);
        cells[i].copyTo(fig(dst));
    }
    char line[256];
    std::snprintf(line, sizeof(line), "frame %d    bars %zu    detect %.2f ms", frameIndex,
                  r.bars.size(), ms);
    cv::putText(fig, line, cv::Point(gap + 2, 24), cv::FONT_HERSHEY_SIMPLEX, 0.72, kWhite, 2,
                cv::LINE_AA);
    return fig;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string videoArg = (argc > 1) ? argv[1] : kDefaultVideo;
    const std::string outDir = (argc > 2) ? argv[2] : "output";
    const std::string path = resolveVideoPath(videoArg, argv[0]);
    const DetectParams params;

    // ---------- 第一遍：读取元数据，统计总帧数与有效帧率 ----------
    int totalFrames = 0;
    double lastPtsMs = 0.0;
    double metaFps = 0.0;
    cv::Size frameSize;
    {
        cv::VideoCapture probe;
        if (!openVideo(path, probe)) {
            std::cerr << "[错误] 无法打开视频: " << path << '\n'
                      << "        请检查路径与文件是否存在，以及 OpenCV 是否启用了 FFmpeg 后端。\n";
            return 1;
        }
        metaFps = probe.get(cv::CAP_PROP_FPS);
        frameSize = cv::Size(static_cast<int>(probe.get(cv::CAP_PROP_FRAME_WIDTH)),
                             static_cast<int>(probe.get(cv::CAP_PROP_FRAME_HEIGHT)));
        cv::Mat tmp;
        while (probe.read(tmp) && !tmp.empty()) {
            ++totalFrames;
            lastPtsMs = probe.get(cv::CAP_PROP_POS_MSEC);
        }
    }

    if (totalFrames == 0 || frameSize.width <= 0 || frameSize.height <= 0) {
        std::cerr << "[错误] 视频中没有可解码的图像帧: " << path << '\n';
        return 1;
    }

    // 该 webm 的帧率元数据不可靠（报 1000fps），因此用“帧数/时间戳跨度”还原真实帧率
    double outFps;
    if (lastPtsMs > 1.0) {
        outFps = totalFrames * 1000.0 / lastPtsMs;
    } else {
        outFps = sanitizeFps(metaFps);
    }
    outFps = std::min(240.0, std::max(1.0, outFps));

    std::error_code ec;
    std::filesystem::create_directories(outDir, ec);
    const std::string base = std::filesystem::path(path).stem().string();

    struct Codec {
        int fourcc;
        const char* ext;
    };
    const Codec codecs[] = {
        {cv::VideoWriter::fourcc('m', 'p', '4', 'v'), ".mp4"},
        {cv::VideoWriter::fourcc('a', 'v', 'c', '1'), ".mp4"},
        {cv::VideoWriter::fourcc('F', 'F', 'V', '1'), ".mkv"},
        {cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), ".avi"},
    };

    cv::VideoWriter writer;
    std::string outVideo;
    // yuv420p 要求宽高为偶数；显式裁掉多出的 1 行/列，避免编码器隐式缩放导致标注位置偏移
    const cv::Size writeSize(frameSize.width & ~1, frameSize.height & ~1);
    if (writeSize.width < 2 || writeSize.height < 2) {
        std::cerr << "[错误] 视频分辨率过小，无法编码: " << frameSize << '\n';
        return 1;
    }
    for (const auto& codec : codecs) {
        const std::string candidate = outDir + "/" + base + "_marked" + codec.ext;
        if (writer.open(candidate, codec.fourcc, outFps, writeSize)) {
            outVideo = candidate;
            break;
        }
        writer.release();
    }
    if (!writer.isOpened()) {
        std::cerr << "[错误] 无法创建输出视频，已尝试 mp4v / avc1 / FFV1 / MJPG。\n";
        return 1;
    }

    std::cout << "输入视频 : " << path << '\n'
              << "分辨率   : " << frameSize.width << "x" << frameSize.height << '\n'
              << "总帧数   : " << totalFrames << "（约 " << lastPtsMs / 1000.0 << " 秒，"
              << outFps << " fps）\n"
              << "元数据帧率: " << metaFps << " fps（不可靠，已用时间戳还原）\n"
              << "输出视频 : " << outVideo << '\n'
              << "输出分辨率: " << writeSize.width << "x" << writeSize.height
              << (writeSize == frameSize ? "" : "（编码器要求偶数尺寸，已显式裁掉 1 行/列）") << '\n'
              << "颜色区间 : H[" << params.hMin << ',' << params.hMax << "] S>=" << params.sMin
              << " V>=" << params.vMin << "，开运算 " << params.openSize << "x" << params.openSize
              << "，长宽比 [" << params.minAspect << ',' << params.maxAspect << "]"
              << "，紧贴度>=" << params.minPurity << "\n\n";

    // ---------- 第二遍：逐帧检测、标注、写回 ----------
    cv::VideoCapture cap;
    if (!openVideo(path, cap)) {
        std::cerr << "[错误] 第二遍打开视频失败: " << path << '\n';
        return 1;
    }

    const double tickFreq = cv::getTickFrequency();
    const std::vector<int> repTargets = {0, totalFrames / 4, totalFrames / 2,
                                         (3 * totalFrames) / 4};
    long long processed = 0, framesWithBars = 0, totalBars = 0;
    double sumMs = 0.0, maxMs = 0.0;
    int maxBars = 0;
    cv::Mat bestOriginal;
    int bestFrameIndex = 0;

    cv::Mat frame;
    while (cap.read(frame) && !frame.empty()) {
        const int idx = static_cast<int>(processed);
        const double t0 = static_cast<double>(cv::getTickCount());
        const FrameResult res = detectBars(frame, params);
        cv::Mat annotated = frame.clone();
        drawBars(annotated, res);
        const double ms =
            (static_cast<double>(cv::getTickCount()) - t0) / tickFreq * 1000.0;
        drawHud(annotated, idx, totalFrames, static_cast<int>(res.bars.size()), ms);

        if (writeSize == frameSize) {
            writer.write(annotated);
        } else {
            writer.write(annotated(cv::Rect(0, 0, writeSize.width, writeSize.height)).clone());
        }

        ++processed;
        sumMs += ms;
        maxMs = std::max(maxMs, ms);
        totalBars += static_cast<long long>(res.bars.size());
        if (!res.bars.empty()) {
            ++framesWithBars;
        }
        if (idx == 0) {
            bestOriginal = frame.clone();
            bestFrameIndex = 0;
        }
        if (static_cast<int>(res.bars.size()) > maxBars) {
            maxBars = static_cast<int>(res.bars.size());
            bestOriginal = frame.clone();
            bestFrameIndex = idx;
        }
        for (int t : repTargets) {
            if (idx == t) {
                char f[512];
                std::snprintf(f, sizeof(f), "%s/%s_frame_%04d_marked.png", outDir.c_str(),
                              base.c_str(), idx);
                cv::imwrite(f, annotated);
            }
        }
    }
    writer.release();
    cap.release();

    // ---------- 代表帧：管线演示图 + 最佳帧标注图 ----------
    if (!bestOriginal.empty()) {
        const double t0 = static_cast<double>(cv::getTickCount());
        const FrameResult bestRes = detectBars(bestOriginal, params);
        const double bestMs =
            (static_cast<double>(cv::getTickCount()) - t0) / tickFreq * 1000.0;

        char f[512];
        std::snprintf(f, sizeof(f), "%s/%s_pipeline_frame_%04d.png", outDir.c_str(),
                      base.c_str(), bestFrameIndex);
        cv::imwrite(f, buildPipelineFigure(bestOriginal, bestRes, bestFrameIndex, bestMs));

        cv::Mat marked = bestOriginal.clone();
        drawBars(marked, bestRes);
        drawHud(marked, bestFrameIndex, totalFrames, static_cast<int>(bestRes.bars.size()), bestMs);
        std::snprintf(f, sizeof(f), "%s/%s_frame_best_%04d_marked.png", outDir.c_str(),
                      base.c_str(), bestFrameIndex);
        cv::imwrite(f, marked);
    }

    // ---------- 统计与异常情况说明 ----------
    std::cout << "处理完成：\n"
              << "  处理帧数      : " << processed << " / " << totalFrames << '\n'
              << "  含灯条的帧数  : " << framesWithBars << '\n'
              << "  无有效灯条帧数: " << (processed - framesWithBars) << '\n'
              << "  灯条总数      : " << totalBars << '\n'
              << "  单帧最多灯条  : " << maxBars << "（第 " << bestFrameIndex << " 帧）\n"
              << "  单帧耗时      : 平均 " << (processed ? sumMs / processed : 0.0)
              << " ms，最大 " << maxMs << " ms\n";

    if (processed < totalFrames) {
        std::cout << "  [提示] 视频流提前结束：预扫描 " << totalFrames << " 帧，实际处理 "
                  << processed << " 帧，源文件可能存在截断或解码错误。\n";
    }
    if (framesWithBars == 0) {
        std::cout << "  [提示] 整段视频未检测到任何有效蓝色灯条，输出仅含原图与帧号标注。\n";
    }
    std::cout << "\n输出文件位于: " << outDir << "/\n";
    return 0;
}
