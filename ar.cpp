#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int dim = 7;
VideoCapture cap("withChessBoard.MOV");

vector<Point3f> cube = {
    Point3f(2, 2, 0),
    Point3f(2, 4, 0),
    Point3f(4, 4, 0),
    Point3f(4, 2, 0),
    Point3f(2, 2, 2),
    Point3f(2, 4, 2),
    Point3f(4, 4, 2),
    Point3f(4, 2, 2)
};
vector<vector<int>> edges = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}
};

int main(int argc, char** argv) {

    // read video frames

    vector<Mat> frames;
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        frames.push_back(frame);
    }

    // camera calibration

    vector<Point2f> corners;
    vector<vector<Point3f>> objps;
    vector<vector<Point2f>> imgps;
    bool found = false;

    vector<Point3f> objp;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            objp.push_back(Point3f(i, j, 0));
        }
    }

    int iteration = frames.size() / 10;

    for (int i = 0; i < 10; i++) {
        Mat frame = frames[i*iteration];
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        found = findChessboardCorners(gray, Size(dim, dim), corners);
        
        if (found) {
            TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1);
            cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1), criteria);

            vector<Point2f> imgp;
            for (int j = 0; j < corners.size(); j++) {
                imgp.push_back(corners[j]);
            }
            objps.push_back(objp);
            imgps.push_back(imgp);
        }
    }

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    calibrateCamera(objps, imgps, Size(frames[0].cols, frames[0].rows), cameraMatrix, distCoeffs, rvecs, tvecs);

    // draw cube on chessboard

    Mat rvec, tvec;
    vector<Point2f> prev_corners;
    for (int i = 0; i < frames.size(); i++) {

        cout << i << endl;

        Mat frame = frames[i];
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        found = findChessboardCorners(gray, Size(dim, dim), corners);

        vector<Point2f> imgp;

        // try solving for vectors instead?
        if (found) {
            TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1);
            cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1), criteria);
            
            for (int j = 0; j < corners.size(); j++) {
                imgp.push_back(corners[j]);
            }

            prev_corners = corners;
        } else { // optical flow from previous frame
            vector<Point2f> next_corners;
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(frames[i - 1], frames[i], prev_corners, next_corners, status, err);

            TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1);
            cornerSubPix(gray, next_corners, Size(5, 5), Size(-1, -1), criteria);

            for (int j = 0; j < next_corners.size(); j++) {
                imgp.push_back(next_corners[j]);
            }

            prev_corners = next_corners;
        }

        solvePnP(objp, imgp, cameraMatrix, distCoeffs, rvec, tvec);

        vector<Point2f> cube2d;
        projectPoints(cube, rvec, tvec, cameraMatrix, distCoeffs, cube2d);

        for (int i = 0; i < edges.size(); i++) {
            Point2f p1 = cube2d[edges[i][0]];
            Point2f p2 = cube2d[edges[i][1]];
            line(frame, p1, p2, Scalar(255, 255, 255), 2);
        }
    }

    // write video frames   

    VideoWriter writer("vr.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(frames[0].cols, frames[0].rows));
    for (int i = 0; i < frames.size(); i++) {
        writer.write(frames[i]);
    }

    return 0;
}