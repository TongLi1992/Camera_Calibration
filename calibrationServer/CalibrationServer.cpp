#include "CalibrationServer.h"
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iterator>


#define BOARD_HEIGHT 9
#define BOARD_WIDTH 6
#define SQUARE_SIZE 0.025
#define ASPECT_RATIO 1
CalibrationServer::CalibrationServer(std::string serverAddr,std::string fileWithID, std::string fileWithoutID, unsigned int maxClients) {
  _serverAddress = serverAddr;
  _maxClients = maxClients;
  _numClients = 0;
  _recordFileWithID = fileWithID; 
  _recordFileWithoutID = fileWithoutID;
}


CalibrationServer::~CalibrationServer() {
  std::cerr << "CalibrationServer destructor" << std::endl;
  _numClients = 0;
  _serverAddress = "";
  _maxClients = 0;
  _recordFileWithID = "";
  _recordFileWithoutID = "";
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
                       grpc::ServerReader<calibration_grpc::Image> *request,
                       calibration_grpc::CameraMatrix *response) {
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (this->_numClients >= this->_maxClients) {
      return grpc::Status::CANCELLED;
    }
    this->_numClients++;
  }
  std::vector<std::vector<cv::Point2f> > imagePoints;
  cv::Mat cameraMatrix, distCoeffs;
  cv::Size imageSize;
  calibration_grpc::Image nextImage;
  while(request->Read(&nextImage)){
    std::cout<<nextImage.messagetype()<<"\n";
    if(nextImage.messagetype() == "QUERY") {
      std::cout<<"Inside QUERY\n";
      std::map<std::string,double> cameraMatrixMap;
      bool found = searchInDB(nextImage.phonemodel(),nextImage.deviceid(), nextImage.capturewidth(), nextImage.captureheight() , cameraMatrixMap);
      if(found) {
        response->set_fx(cameraMatrixMap.find("fx")->second);
        response->set_fy(cameraMatrixMap.find("fy")->second);
        response->set_cx(cameraMatrixMap.find("cx")->second);
        response->set_cy(cameraMatrixMap.find("cy")->second);
        response->set_resultmessage("Succeed");
      } else {
        response->set_resultmessage("Model did not found");
      }
      this->_numClients--;  
      return grpc::Status::OK;
    } else {
      std::vector<uchar> data(nextImage.image().begin(), nextImage.image().end());
      assert(data.size() > 0);
      bool copyData = false;
      cv::Mat view = imdecode(cv::Mat(data, copyData), cv::IMREAD_GRAYSCALE);
      //imwrite("image"+std::to_string(i) +".jpg", view);
      imageSize = view.size();
      
      //find_pattern
      std::vector<cv::Point2f> pointBuf;
      bool found;
      int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
      chessBoardFlags |= cv::CALIB_CB_FAST_CHECK;
      found = findChessboardCorners( view, cv::Size(BOARD_WIDTH, BOARD_HEIGHT), pointBuf, chessBoardFlags);
      if ( found) {
        cornerSubPix( view, pointBuf, cv::Size(11,11),
          cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1 ));
        imagePoints.push_back(pointBuf);
      }
    }
   
  }
  /*std::cout<<"ImagePoints length is "<<imagePoints.size()<<std::endl;
  for(int k = 0; k < imagePoints.size(); k++) {
    std::cout<<"ImagePoint "<<k<<"  size is " <<imagePoints[k].size()<<"\n";
    for(auto point: imagePoints[k]) {
        std::cout<<point<<" ";
    }
    std::cout<<std::endl;
  }*/
  try {
    runCalibrationAndSave(imageSize,  cameraMatrix, distCoeffs, imagePoints);
  } catch(cv::Exception e) {
    response->set_resultmessage("Invalid");
    this->_numClients--;
    return grpc::Status::OK;
  }
  response->set_fx(cameraMatrix.at<double>(0,0));
  response->set_fy(cameraMatrix.at<double>(1,1));
  response->set_cx(cameraMatrix.at<double>(0,2));
  response->set_cy(cameraMatrix.at<double>(1,2));
  response->set_resultmessage("Succeed");
  this->_numClients--;  
  return grpc::Status::OK;
}

bool CalibrationServer::runCalibrationAndSave(cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                                          std::vector<std::vector<cv::Point2f> > imagePoints) {
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<float> reprojErrs;
  double totalAvgErr = 0;
  bool ok = runCalibration(imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs,totalAvgErr);
  std::cout << (ok ? "Calibration succeeded" : "Calibration failed")
       << ". avg re projection error = " << totalAvgErr << std::endl;
  std::cout<<"The cameraMatrix is :"<<cameraMatrix<<std::endl;
  return ok;
}

bool CalibrationServer::runCalibration(cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                            std::vector<std::vector<cv::Point2f> > imagePoints, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs,
                            std::vector<float>& reprojErrs,  double& totalAvgErr) {
  cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
  cameraMatrix.at<double>(0,0) = ASPECT_RATIO;
  distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
  std::vector<std::vector<cv::Point3f> > objectPoints(1);
  calcBoardCornerPositions(objectPoints[0]);
  objectPoints.resize(imagePoints.size(),objectPoints[0]);

  //Find intrinsic and extrinsic camera parameters
  double rms;
  rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, cv::CALIB_FIX_K4|cv::CALIB_FIX_K5);
  std::cout << "Re-projection error reported by calibrateCamera: "<< rms << std::endl;
  bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
  totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
                                          distCoeffs, reprojErrs);
  return ok;
}

void CalibrationServer::calcBoardCornerPositions(std::vector<cv::Point3f>& corners) {
  corners.clear();
  for( int i = 0; i < BOARD_HEIGHT; ++i ) {
    for( int j = 0; j < BOARD_WIDTH; ++j ) {
      corners.push_back(cv::Point3f((float)j*SQUARE_SIZE,(float)i*SQUARE_SIZE, 0));
    }
  }
}
            
double CalibrationServer::computeReprojectionErrors( const std::vector<std::vector<cv::Point3f> >& objectPoints,
                                         const std::vector<std::vector<cv::Point2f> >& imagePoints,
                                         const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                                         const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs,
                                         std::vector<float>& perViewErrors) {
  std::vector<cv::Point2f> imagePoints2;
  size_t totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize(objectPoints.size());
  for(size_t i = 0; i < objectPoints.size(); ++i ) {
    projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
    err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);
    size_t n = objectPoints[i].size();
    perViewErrors[i] = (float) std::sqrt(err*err/n);
    totalErr        += err*err;
    totalPoints     += n;
  }
  return std::sqrt(totalErr/totalPoints);
}

bool CalibrationServer::searchInDB(std::string model, std::string deviceId, int captureWidth, int captureHeight,  std::map<std::string,double> &cameraMatrix) {
  std::ifstream fin(_recordFileWithID);
  if(!fin) {
    std::cout<<"_recordFileWithID opening failed\n";
    return false;
  }
  while(!fin.eof()) {
    std::string line;
    std::getline(fin, line);
    std::vector<std::string> tokens = split(line, ',');
    if (model == tokens[0] && deviceId == tokens[1] && 
        std::to_string(captureWidth) == tokens[2] &&
        std::to_string(captureHeight) == tokens[3]) {
      cameraMatrix.insert(std::pair<std::string,double>("fx",std::stod(tokens[4])));
      cameraMatrix.insert(std::pair<std::string,double>("fy",std::stod(tokens[5])));
      cameraMatrix.insert(std::pair<std::string,double>("cx",std::stod(tokens[6])));
      cameraMatrix.insert(std::pair<std::string,double>("cy",std::stod(tokens[7])));
      fin.close();
      return true;
    }
  }
  fin.close();
  fin.open(_recordFileWithoutID);
  if(!fin) {
    std::cout<<"_recordFileWithoutID opening failed\n";
    return false;
  }
  while(!fin.eof()) {
    std::string line;
    std::getline(fin, line);
    std::vector<std::string> tokens = split(line, ',');
    if (model == tokens[0] && 
        std::to_string(captureWidth) == tokens[1] &&
        std::to_string(captureHeight) == tokens[2]) {
      cameraMatrix.insert(std::pair<std::string,double>("fx",std::stod(tokens[3])));
      cameraMatrix.insert(std::pair<std::string,double>("fy",std::stod(tokens[4])));
      cameraMatrix.insert(std::pair<std::string,double>("cx",std::stod(tokens[5])));
      cameraMatrix.insert(std::pair<std::string,double>("cy",std::stod(tokens[6])));
      fin.close();
      return true;
    }
  }
  fin.close();
  return false;

  // fout<<model<<","<<deviceId<<","<<captureWidth<<","<<captureHeight<<","<<"1 2 3 4\n";
}

std::vector<std::string> CalibrationServer::split(const std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    result.push_back(item);
  }
  return result;
}

