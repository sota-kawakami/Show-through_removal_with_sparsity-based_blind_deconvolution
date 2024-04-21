#pragma once

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Member.h"



void SetImageToMatrix(Eigen::MatrixXf& mat, std::string in_name, const Member& mem, const int tLY, const int tLX);
void SaveImage(const Eigen::MatrixXf& mat, std::string in_name, const Member& mem);

Eigen::VectorXf MatrixToVector(const Eigen::MatrixXf& mat);
Eigen::MatrixXf VectortoMatrix(const Eigen::VectorXf& vec, const int height, const int width);

Eigen::MatrixXf MakeGaussianBlurKernel( const float r, const int KY, const int KX);
void Convolution(const Eigen::MatrixXf& x_mat, Eigen::MatrixXf& y_mat, const Eigen::MatrixXf& w_mat, const Member& mem);

Eigen::MatrixXf horizontal_flip(const Eigen::MatrixXf image);
Eigen::MatrixXf show_througth(const Eigen::MatrixXf omote, const Eigen::MatrixXf ura, const double alpha);

float CalculatePSNR(const Eigen::VectorXf correct_data, const  Eigen::VectorXf estimated_data, float max);
float CalculateSSIM(const Eigen::MatrixXf correct_data, const  Eigen::MatrixXf estimated_data);
Eigen::MatrixXf RoundOffMatrix(const Eigen::MatrixXf& mat);

Eigen::MatrixXf PaddingMartix(const Eigen::MatrixXf& input, const int padding_length);

