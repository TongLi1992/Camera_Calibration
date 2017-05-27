#include "CalibrationServer.h"
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

CalibrationServer::CalibrationServer(int serverAddr, unsigned int maxClients) {
  _serverAddress = std::to_string(serverAddr);
  _maxClients = maxClients;
  _numClients = 0;
}


CalibrationServer::~CalibrationServer() {
  std::cerr << "CalibrationServer destructor" << std::endl;
  _numClients = 0;
  _serverAddress = "";
  _maxClients = 0;
}

void CalibrationServer::RunServer() {
  std::cerr << "DEBUG: CalibrationServer start()" << std::endl;
  std::string server_address(_serverAddress);
  grpc::ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort("0.0.0.0:"+server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(this);
  // Finally assemble the server.
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

grpc::Status CalibrationServer::calibrate(grpc::ServerContext *context,
                       const calibration_grpc::Images *request,
                       calibration_grpc::CameraMatrix *response) {
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (this->_numClients >= this->_maxClients) {
      response->set_foundname("Too many clients, server is busy");
      return grpc::Status::OK;
    }
    this->_numClients++;
  }

  vector<vector<Point2f> > imagePoints;
  Mat cameraMatrix, distCoeffs;
  Size imageSize;
  for(int i = 0; i < request->image_size(); i++) {
    std::vector<uchar> data(request->image(i).begin(), request->image(i).end());
    assert(data.size() > 0);
    bool copyData = false;
    cv::Mat view = imdecode(cv::Mat(data, copyData), cv::IMREAD_GRAYSCALE);
    imageSize = view.size();
    
    //find_pattern
    vector<Point2f> pointBuf;
    bool found;
    int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
    chessBoardFlags |= CALIB_CB_FAST_CHECK;
    found = findChessboardCorners( view, s.boardSize, pointBuf, chessBoardFlags);
    if ( found) {
      cornerSubPix( viewGray, pointBuf, Size(11,11),
        Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ));
      imagePoints.push_back(pointBuf);
    }
  }
  runCalibrationAndSave(imageSize,  cameraMatrix, distCoeffs, imagePoints);
  response.set_fx(cameraMatrix.at<double>(0,0));
  response.set_fy(cameraMatrix.at<double>(1,1));
  response.set_cx(cameraMatrix.at<double>(0,2));
  response.set_cy(cameraMatrix.at<double>(1,2));
  this->_numClients--;  
  return grpc::Status::OK;
}

bool CalibrationServer::runCalibrationAndSave(Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                                          vector<vector<Point2f> > imagePoints) {
  vector<Mat> rvecs, tvecs;
  vector<float> reprojErrs;
  double totalAvgErr = 0;
  bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs,totalAvgErr);
  cout << (ok ? "Calibration succeeded" : "Calibration failed")
       << ". avg re projection error = " << totalAvgErr << endl;
  std::cout<<"The cameraMatrix is :"<<cameraMatrix<<std::endl;
  return ok;
}

bool CalibrationServer::runCalibration( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                            vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
                            vector<float>& reprojErrs,  double& totalAvgErr) {
  cameraMatrix = Mat::eye(3, 3, CV_64F);
  cameraMatrix.at<double>(0,0) = CalibrationServer::ASPECT_RATIO;
  distCoeffs = Mat::zeros(8, 1, CV_64F);
  vector<vector<Point3f> > objectPoints(1);
  calcBoardCornerPositions(objectPoints[0]);
  objectPoints.resize(imagePoints.size(),objectPoints[0]);

  //Find intrinsic and extrinsic camera parameters
  double rms;
  rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,s.flag);
  cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;
  bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
  totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
                                          distCoeffs, reprojErrs, s.useFisheye);
  return ok;
}

void CalibrationServer::calcBoardCornerPositions(vector<Point3f>& corners) {
  corners.clear();
  for( int i = 0; i < BOARD_HEIGHT; ++i ) {
    for( int j = 0; j < BOARD_WIDTH; ++j ) {
      corners.push_back(Point3f(j*SQUARE_SIZE, i*SQUARE_SIZE, 0));
    }
  }
}
            
double CalibrationServer::computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                         const vector<vector<cv::Point2f> >& imagePoints,
                                         const vector<cv::Mat>& rvecs, const vector<cv::Mat>& tvecs,
                                         const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs,
                                         vector<float>& perViewErrors) {
  vector<cv::Point2f> imagePoints2;
  size_t totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize(objectPoints.size());
  for(size_t i = 0; i < objectPoints.size(); ++i ) {
    projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
    err = norm(imagePoints[i], imagePoints2, NORM_L2);
    size_t n = objectPoints[i].size();
    perViewErrors[i] = (float) std::sqrt(err*err/n);
    totalErr        += err*err;
    totalPoints     += n;
  }
  return std::sqrt(totalErr/totalPoints);
}
