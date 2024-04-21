#include "Member.h"
#include <Shlwapi.h>
#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>

#pragma comment(lib, "Shlwapi.lib")



using namespace std;

#include "nlohmann/json.hpp"
using json = nlohmann::json;



Member::Member(string input_json)
{

	if (!PathFileExistsA(input_json.c_str())) {
		cerr << "can't read" << input_json <<  endl;
		exit(1);
	}

	std::ifstream reading(input_json, std::ios::in);
	json j;
	reading >> j;

	mode = j["mode"];
	num_of_input_images = j["num_of_input_images"];

	method = j["method"];


	if (mode == "simulation") {
		for (int i = 0; i < j["original_image_name"].size(); i++) {
			original_image_name.push_back(j["original_image_name"][i]);
			cout << "input original image:" << original_image_name[i] << endl;
		}
		before_convolution_image_height = j["before_convolution_image_height"];
		before_convolution_image_width = j["before_convolution_image_width"];
		real_kernel_height = j["real_kernel_height"];
		real_kernel_width = j["real_kernel_width"];
		correct_alpha_rate = j["alpha_rate"];
		image_width = before_convolution_image_width - (real_kernel_width - 1);
		image_height = before_convolution_image_height - (real_kernel_height - 1);
		num_of_kernels = 1;

		for (int i = 0; i < j["gaussian_params"].size(); i++)
			gaussian_params.push_back(j["gaussian_params"][i]);
	}
	else {
		for (int i = 0; i < j["input_degraded_image"].size(); i++) {
			degraded_image_name.push_back(j["input_degraded_image"][i]);
			cout << "input degraded image:" << degraded_image_name[i] << endl;
		}

		image_width = j["image_width"];
		image_height = j["image_height"];
	}
		
	estimated_kernel_height = j["estimated_kernel_height"];
	estimated_kernel_width = j["estimated_kernel_width"];

	step_size = j["step_size"];

	if (method == "minrank") {
		prox_mode = j["prox_mode"];
		threshold_method = j["threshold_method"];
		if (threshold_method == "soft" || threshold_method == "ST") {
			soft_thresh_val = j["soft_thresh_val"];
			nuclear_norm_param = j["nuclear_norm_param"];
		}
		
		smoothing_term_param = j["smoothing_term_param"];
		enable_u8_clamp = j["enable_u8_clamp"];
	}
	else if (method == "sharma") {
		back_pixel_val = j["back_pixel_val"];
		kernel_l = j["kernel_l"];
		threshold = j["sharma_threshold"];

	}
	else if (method == "nmf") {
		back_pixel_val = j["back_pixel_val"];

	}
	add_title = j["add_title"];
	save_img = j["save_img"];
	save_img_format = j["save_img_format"];
	img_save_iteration = j["img_save_iteration"];
	log_save_iteration = j["log_save_iteration"];
	num_of_iteration = j["num_of_iteration"];
	enable_input_round_off = j["enable_input_round_off"];
	enable_output_round_off = j["enable_output_round_off"];


	image_size = image_height * image_width;
	estimated_kernel_size = estimated_kernel_height * estimated_kernel_width;
	estimated_kernel_half_width = (estimated_kernel_width - 1) / 2;
	estimated_kernel_half_height = (estimated_kernel_height - 1) / 2;

}