#pragma once
#include <atomic>
#include <mutex>  
#include "CalibrationService.grpc.pb.h"
#include <grpc++/grpc++.h>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


class CalibrationServer final : public calibration_grpc::CalibrationService::Service {

public:
  explicit CalibrationServer(std::string serverAddr, unsigned int maxClients);
  ~CalibrationServer();

  void RunServer();
  grpc::Status calibrate(grpc::ServerContext *context,
                         grpc::ServerReader<calibration_grpc::Image> *request,
                         calibration_grpc::CameraMatrix *response);
private:
  bool runCalibrationAndSave(cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                                            std::vector<std::vector<cv::Point2f> > imagePoints);
  bool runCalibration( cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                       std::vector<std::vector<cv::Point2f> > imagePoints, std::vector<cv::Mat>& rvecs, 
                       std::vector<cv::Mat>&tvecs, std::vector<float>& reprojErrs,  double& totalAvgErr);
  void calcBoardCornerPositions(std::vector<cv::Point3f>& corners);
  double computeReprojectionErrors( const std::vector<std::vector<cv::Point3f> >& objectPoints,
                                    const std::vector<std::vector<cv::Point2f> >& imagePoints,
                                    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                                    const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs,
                                    std::vector<float>& perViewErrors);
private:
  std::string _serverAddress;
  std::atomic<unsigned int> _numClients;
  std::atomic<unsigned int> _maxClients; 
  std::mutex _mutex;
};
