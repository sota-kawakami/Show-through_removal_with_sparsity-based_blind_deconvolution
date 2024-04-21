#include "matrix_tool.h"
#include "Member.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <regex>
#include <windows.h>
#include <fstream>
#include <iostream>

using namespace Eigen;
using namespace std;

cv::Mat eigin_to_cv_mat(const MatrixXf& mat) {
	cv::Mat cv_mat = cv::Mat::zeros(cv::Size(mat.cols(), mat.rows()), CV_8U);
	for (int i = 0; i < mat.rows(); i++) {
		for (int j = 0; j < mat.cols(); j++) {
			if (mat(i, j) > 255)
				cv_mat.at<unsigned char>(i, j) = 255;
			else if (mat(i, j) < 0)
				cv_mat.at<unsigned char>(i, j) = 0;
			else
				cv_mat.at<unsigned char>(i, j) = round(mat(i, j));
		}
	}
	return cv_mat;
}

cv::Mat eigin_to_cv_vec(const VectorXf& vec) {
	cv::Mat cv_vec = cv::Mat::zeros(cv::Size(vec.size(),1), CV_8U);
	for (int i = 0; i < vec.size(); i++) {
		if (vec(i) > 255)
			cv_vec.at<unsigned char>(i, 0) = 255;
		else if (vec(i) < 0)
			cv_vec.at<unsigned char>(i, 0) = 0;
		else
			cv_vec.at<unsigned char>(i, 0) = round(vec(i));
	}
	return cv_vec;
}

void SetImageToMatrix(MatrixXf& mat, string image_name, const Member &mem, const int input_image_height, const int input_image_width) {


	cv::Mat	cv_mat = cv::imread(image_name);
	for (int i = 0; i < input_image_height; i++) {
		for (int j = 0; j < input_image_width; j++) {
			mat(i, j) = cv_mat.at<cv::Vec3b>(i, j)[0];
		}
	}

}


void SaveImage(const MatrixXf& mat, string input_image_name, const Member &mem) {
	if (!mem.save_img)
		return;


	const char* now_floder_name = mem.folder_name.c_str();
	const char* image_name_char = input_image_name.c_str();
	string folder_and_image_name = now_floder_name + string("\\") + image_name_char;

	const int input_image_width = mat.cols();
	const int input_image_height = mat.rows();


	if (mem.save_img_format == "bmp" || mem.save_img_format == "png") {

		cv::Mat cv_mat = eigin_to_cv_mat(mat);

		if (mem.save_img_format == "bmp")
			cv::imwrite(folder_and_image_name + ".bmp", cv_mat);
		if (mem.save_img_format == "png")
			cv::imwrite(folder_and_image_name + ".png", cv_mat);
	}
	else {
		cout << mem.save_img_format << "is a nonexistent format" << endl;
	}


}


VectorXf MatrixToVector(const MatrixXf& mat) {

	const int width = mat.cols();
	const int height = mat.rows();

	MatrixXf vec = VectorXf::Zero(height*width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			vec(i * width + j) = mat(i, j);
		}
	}

	return vec;

}

MatrixXf VectortoMatrix(const VectorXf& vec, const int height, const int width) {

	MatrixXf mat = MatrixXf::Ones(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			mat(i, j) = vec(i * width + j);
		}
	}

	return mat;

}


MatrixXf RoundOffMatrix(const MatrixXf& mat) {

	MatrixXf buf_mat = MatrixXf::Zero(mat.rows(), mat.cols());
	
	for (int i = 0; i <mat.rows(); i++) {
		for (int j = 0; j < mat.cols(); j++) {
			buf_mat(i, j) =  round(mat(i,j));
		}
	}

	return buf_mat;

}



double getSSIM(const cv::Mat& i1, const cv::Mat& i2)
{
	const double C1 = 6.5025, C2 = 58.5225;
	int d = CV_32F;
	cv::Mat I1, I2;
	i1.convertTo(I1, d);
	i2.convertTo(I2, d);
	cv::Mat I1_2 = I1.mul(I1);
	cv::Mat I2_2 = I2.mul(I2);
	cv::Mat I1_I2 = I1.mul(I2);
	cv::Mat mu1, mu2;
	GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
	cv::Mat mu1_2 = mu1.mul(mu1);
	cv::Mat mu2_2 = mu2.mul(mu2);
	cv::Mat mu1_mu2 = mu1.mul(mu2);
	cv::Mat sigma1_2, sigam2_2, sigam12;
	GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	GaussianBlur(I2_2, sigam2_2, cv::Size(11, 11), 1.5);
	sigam2_2 -= mu2_2;

	GaussianBlur(I1_I2, sigam12, cv::Size(11, 11), 1.5);
	sigam12 -= mu1_mu2;
	cv::Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigam12 + C2;
	t3 = t1.mul(t2);

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigam2_2 + C2;
	t1 = t1.mul(t2);

	cv::Mat ssim_map;
	divide(t3, t1, ssim_map);
	cv::Scalar mssim = mean(ssim_map);

	double ssim = mssim.val[0];

	return ssim;
}

float CalculateSSIM(const MatrixXf correct_data, const  MatrixXf estimated_data) {

	cv::Mat cv_correct_mat = eigin_to_cv_mat(correct_data);
	cv::Mat cv_estimated_mat = eigin_to_cv_mat(estimated_data);


	return getSSIM(cv_correct_mat, cv_estimated_mat);
}


MatrixXf MakeGaussianBlurKernel(const float gaussian_param, const int kernel_height, const int kernel_width) {

	MatrixXf kernel_mat = MatrixXf::Zero(kernel_height, kernel_width);

	const float pi = 3.141592653589793f;
	const int kernel_half_width = (kernel_width - 1) / 2;
	const int kernel_half_height = (kernel_height - 1) / 2;

	for (int i = 0; i < kernel_height; i++) {
		for (int j = 0; j < kernel_width; j++) {
			int x = abs(j - kernel_half_width);
			int y = abs(i - kernel_half_height);
			kernel_mat(i, j) = (float)((1 / (2.0*pi*gaussian_param*gaussian_param))*exp(-((x*x + y*y) / (2 * gaussian_param*gaussian_param))));
		}
	}

	return kernel_mat /= kernel_mat.sum();


}

void Convolution(const MatrixXf& input_image_mat, MatrixXf& blurred_image_mat, const MatrixXf& kernel_mat, const Member &mem) {

	const int real_kernel_half_width = (mem.real_kernel_width - 1) / 2;
	const int real_kernel_half_height = (mem.real_kernel_height - 1) / 2;

	int before_convolution_height, before_convolution_width;



	before_convolution_height = mem.before_convolution_image_height;
	before_convolution_width = mem.before_convolution_image_width;


	VectorXf input_image_vec = MatrixToVector(input_image_mat  );
	VectorXf kernel_vec = MatrixToVector(kernel_mat);

	MatrixXf X = input_image_vec*kernel_vec.transpose();


	for (int i = 0; i < before_convolution_height; i++) {
		for (int j = 0; j < before_convolution_width; j++) {
			blurred_image_mat(i, j) = 0;
			for (int h = 0; h < mem.real_kernel_height; h++) {
				for (int k = 0; k < mem.real_kernel_width; k++) {
					if (j - real_kernel_half_width + k >= 0 && i - real_kernel_half_height + h >= 0 && i - real_kernel_half_height + h < before_convolution_height && j - real_kernel_half_width + k < before_convolution_width) {
						blurred_image_mat(i, j) += X((i - real_kernel_half_height + h)* before_convolution_width + (j - real_kernel_half_width + k), h*mem.real_kernel_width + k);
					}
					else {
						blurred_image_mat(i, j) += input_image_mat(i, j)   *   kernel_vec(h*mem.real_kernel_width + k);
					}
				}
			}

		}
	}


}



MatrixXf horizontal_flip(const MatrixXf image) {
	MatrixXf image_buf = MatrixXf::Zero(image.rows(), image.cols());

	for (int i = 0; i < image.cols(); i++) {
		image_buf.block(0, i, image.rows(), 1) = image.block(0, image.cols() - i - 1, image.rows(), 1);
	}
	return image_buf;
}

Eigen::MatrixXf show_througth(const Eigen::MatrixXf front,const Eigen::MatrixXf back, const double alpha) {

	MatrixXf image_buf = front;

	for (int i = 0; i < image_buf.rows(); i++) {
		for (int j = 0; j < image_buf.cols(); j++) {

			image_buf(i, j) = front(i, j) - alpha * (255 - back(i, j));


			if (image_buf(i, j) < 0) image_buf(i, j) = 0;
			if (image_buf(i, j) > 255) image_buf(i, j) = 255;

		}
	}
	return image_buf;

}

float CalculatePSNR(const VectorXf correct_data, const  VectorXf estimated_data, float max) {


	cv::Mat cv_correct_mat = eigin_to_cv_mat(correct_data);
	cv::Mat cv_estimated_mat = eigin_to_cv_mat(estimated_data);

	return cv::PSNR(cv_correct_mat, cv_estimated_mat, max);


}





Eigen::MatrixXf PaddingMartix(const Eigen::MatrixXf& input, const int pad_l) {
	cv::Mat cv_mat = eigin_to_cv_mat(input);

	cv::Mat cv_out_mat = cv::Mat::zeros(cv::Size(input.cols()+ 2* pad_l, input.rows() + 2 * pad_l), CV_8U);
	copyMakeBorder(cv_mat, cv_out_mat, pad_l, pad_l, pad_l, pad_l, cv::BORDER_REPLICATE);


	MatrixXf out_mat = MatrixXf::Zero(input.rows() + 2 * pad_l, input.cols() + 2 * pad_l);
	for (int i = 0; i < out_mat.rows(); i++) {
		for (int j = 0; j < out_mat.cols(); j++) {

			out_mat(i, j) = cv_out_mat.at<unsigned char>(i, j);

		}
	}

	return out_mat;
	

}