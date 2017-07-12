#include "calibrationServer/CalibrationServer.h"
const int MAX_CLIENTS = 10;
int main(int argc, char *argv[]) {
  CalibrationServer server(argv[1], argv[2], argv[3],  MAX_CLIENTS);
  server.RunServer();
}
