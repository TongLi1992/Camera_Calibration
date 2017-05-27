#pragma once
#include <atomic>
#include <mutex>  
#include "CalibrationService.grpc.pb.h"
#include <grpc++/grpc++.h>

class CalibrationServer final : public calibration_grpc::CalibrationService::Service {

public:
  explicit CalibrationServer(int serverAddr, unsigned int maxClients);
  ~CalibrationServer();

  void RunServer();
  grpc::Status calibrate(grpc::ServerContext *context,
                         const calibration_grpc::Images *request,
                         calibration_grpc::CameraMatrix *response);
private:
  std::vector<double> runCalibrationAndSave(Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                                            vector<vector<Point2f> > imagePoints);
  bool runCalibration( Size& imageSize, Mat& cameraMatrix, Mat& distt& distCoeffs,
                       vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, 
                       vector<Mat>&tvecs, vector<float>& reprojErrs,  double& totalAvgErr);
  void calcBoardCornerPositions(vector<Point3f>& corners);
  double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                    const vector<vector<Point2f> >& imagePoints,
                                    const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                    const Mat& cameraMatrix , const Mat& distCoeffs,
                                    vector<float>& perViewErrors);
private:
  std::string _serverAddress;
  std::atomic<unsigned int> _numClients;
  std::atomic<unsigned int> _maxClients; 
  std::mutex _mutex;
};
