#pragma once

#include <iostream>
#include <vector>
#include <time.h>
#include "Eigen/Core"
#include "Eigen/Dense"

class Member
{
public:
	Member(std::string input_json);
	std::string folder_name;
	std::string add_title;
	std::vector<std::string> original_image_name;
	std::vector<std::string> degraded_image_name;

	int image_width;
	int image_height;
	int image_size;

	int pad_image_width;
	int pad_image_height;
	int pad_image_size;

	int before_convolution_image_width;
	int before_convolution_image_height;

	int estimated_kernel_width;
	int estimated_kernel_height;

	int estimated_kernel_half_width;
	int estimated_kernel_half_height;
	int estimated_kernel_size;
	int real_kernel_width;
	int real_kernel_height;
	int real_kernel_size;
	std::vector<float> gaussian_params;

	int num_of_input_images;
	int num_of_kernels;
	int num_of_iteration;

	int img_save_iteration;
	int log_save_iteration;

	int save_img;
	std::string save_img_format;

	float correct_alpha_rate;


	std::string mode;
	std::string method;

	int enable_input_round_off;
	int enable_output_round_off;




	std::vector<Eigen::MatrixXf> input_image_mats;

	std::vector<Eigen::VectorXf> estimated_image_vecs;

	std::vector<Eigen::VectorXf> correct_image_vecs;
	

	std::vector<Eigen::VectorXf> estimated_kernel_vecs;
	std::vector<Eigen::VectorXf> correct_kernel_vecs;


	int iter;

	//min rank
	Eigen::MatrixXf X;
	std::string prox_mode;
	std::string threshold_method;
	float soft_thresh_val;
	float date_norm_param;
	float nuclear_norm_param;
	float smoothing_term_param;
	float step_size;
	float kernel_sparse_alpha;
	float smoosing_alpha;
	int enable_u8_clamp;



	//sharma
	int back_pixel_val;
	int kernel_l;
	float threshold;


	clock_t start;
	clock_t end;
	double time;



};