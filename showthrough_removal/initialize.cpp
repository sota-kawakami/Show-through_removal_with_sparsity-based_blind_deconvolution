#include "initialize.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Member.h"
#include "matrix_tool.h"
#include "mytool.h"

#include <vector>

#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;


void MakeKernelAndBlur(Member& mem) {


	cout << "image num " << mem.num_of_input_images << endl;
	cout << "input image size (before conv) " << mem.before_convolution_image_height << "x" << mem.before_convolution_image_width << endl;
	cout << "kernel size " << mem.real_kernel_height << "x" << mem.real_kernel_width << endl;
	cout << "esti kernel size " << mem.estimated_kernel_height << "x" << mem.estimated_kernel_width << endl;


	if (mem.mode == "simulation") {

		MatrixXf kernel_mat = MakeGaussianBlurKernel(mem.gaussian_params[0], mem.real_kernel_height, mem.real_kernel_width);

		const int real_kernel_half_width = (int)(mem.real_kernel_width - 1) / 2;
		MatrixXf before_convoluted_input_image_mat1 = MatrixXf::Zero(mem.before_convolution_image_height, mem.before_convolution_image_width);
		MatrixXf before_convoluted_input_image_mat2 = MatrixXf::Zero(mem.before_convolution_image_height, mem.before_convolution_image_width);
		SetImageToMatrix(before_convoluted_input_image_mat1, mem.original_image_name[0], mem, mem.before_convolution_image_height, mem.before_convolution_image_width);
		SaveImage(before_convoluted_input_image_mat1, "original1", mem);
		SetImageToMatrix(before_convoluted_input_image_mat2, mem.original_image_name[1], mem, mem.before_convolution_image_height, mem.before_convolution_image_width);
		SaveImage(before_convoluted_input_image_mat2, "original2", mem);


		MatrixXf input_image_mat1 = before_convoluted_input_image_mat1.block(real_kernel_half_width, real_kernel_half_width, mem.image_height, mem.image_width);
		MatrixXf pad_in1_pad = PaddingMartix(input_image_mat1, mem.estimated_kernel_half_height);
		SaveImage(input_image_mat1, "in0_" + to_string(mem.image_height) + "x" + to_string(mem.image_width) , mem);
		mem.correct_image_vecs.push_back(MatrixToVector(input_image_mat1));


		MatrixXf input_image_mat2 = before_convoluted_input_image_mat2.block(real_kernel_half_width, real_kernel_half_width, mem.image_height, mem.image_width);
		MatrixXf pad_in2_pad = PaddingMartix(input_image_mat2, mem.estimated_kernel_half_height);
		SaveImage(input_image_mat2, "in1_" + to_string(mem.image_height) + "x" + to_string(mem.image_width) , mem);
		mem.correct_image_vecs.push_back(MatrixToVector(input_image_mat2));

		MatrixXf blurred_image_mat1 = MatrixXf::Zero(mem.before_convolution_image_height, mem.before_convolution_image_width);
		MatrixXf blurred_image_mat2 = MatrixXf::Zero(mem.before_convolution_image_height, mem.before_convolution_image_width);

		Convolution(before_convoluted_input_image_mat1, blurred_image_mat1, kernel_mat, mem);
		MatrixXf convoluted_image_without_frame_mat1 = blurred_image_mat1.block(real_kernel_half_width, real_kernel_half_width, mem.image_height, mem.image_width);
		Convolution(before_convoluted_input_image_mat2, blurred_image_mat2, kernel_mat, mem);
		MatrixXf convoluted_image_without_frame_mat2 = blurred_image_mat2.block(real_kernel_half_width, real_kernel_half_width, mem.image_height, mem.image_width);


		MatrixXf in1 = show_througth(input_image_mat1, horizontal_flip(convoluted_image_without_frame_mat2), mem.correct_alpha_rate);
		MatrixXf in2 = show_througth(input_image_mat2, horizontal_flip(convoluted_image_without_frame_mat1), mem.correct_alpha_rate);
		SaveImage(in1, "in_ss_0", mem);
		SaveImage(in2, "in_ss_1", mem);

		MatrixXf in1_pad = PaddingMartix(in1, mem.estimated_kernel_half_height);
		MatrixXf in2_pad = PaddingMartix(in2, mem.estimated_kernel_half_height);


		mem.input_image_mats.push_back(in1_pad);
		mem.input_image_mats.push_back(in2_pad);


		mem.pad_image_height = mem.image_height + mem.estimated_kernel_half_height * 2;
		mem.pad_image_width = mem.image_width + mem.estimated_kernel_half_height * 2;
		mem.pad_image_size = mem.pad_image_height * mem.pad_image_width;


		SaveImage(mem.input_image_mats[0], "in_ss_pad_0", mem);
		SaveImage(mem.input_image_mats[1], "in_ss_pad_1", mem);


		const int difference_kernel_width = (mem.estimated_kernel_width - 1) / 2 - (mem.real_kernel_width - 1) / 2;
		const int difference_kernel_height = (mem.estimated_kernel_height - 1) / 2 - (mem.real_kernel_height - 1) / 2;

		mem.correct_kernel_vecs.resize(1);

		MatrixXf learge_real_kernel_mat = MatrixXf::Zero(mem.estimated_kernel_height, mem.estimated_kernel_width);
		learge_real_kernel_mat.block(difference_kernel_height, difference_kernel_width, mem.real_kernel_height, mem.real_kernel_width) = kernel_mat;
		mem.correct_kernel_vecs[0] = MatrixToVector(learge_real_kernel_mat);


		string log_text_namt = mem.folder_name;
		log_text_namt += "//true_psf.txt";

		ofstream fs1(log_text_namt, ios::app);
		if (fs1.fail())
			cout << "can't read txt" << endl;
		for (int i = 0; i < mem.estimated_kernel_height; i++) {
			for (int j = 0; j < mem.estimated_kernel_width; j++) {
				fs1 << to_string(learge_real_kernel_mat(j, i)) << " ";
			}
			fs1 << endl;
		}

		fs1.close();
	}


	cout << "input image size (aftere conv) " << mem.image_height << "x" << mem.image_width << endl;
	cout << "input image size (aftere conv, pad) " << mem.pad_image_height << "x" << mem.pad_image_width << endl;

}

void Initialize(Member& mem) {

	const int estimated_kernel_size = mem.estimated_kernel_height * mem.estimated_kernel_width;


	VectorXf kernel_vec;
	kernel_vec = VectorXf::Ones(estimated_kernel_size);
	kernel_vec /= kernel_vec.sum();
	VectorXf kernel_vec2 = VectorXf::Ones(estimated_kernel_size + 1);
	kernel_vec2.block(0, 0, kernel_vec.size(), 1) = kernel_vec*0.5;
	mem.estimated_kernel_vecs.push_back(kernel_vec2);




	if (mem.mode == "real" ) {

		for (int i = 0; i < mem.num_of_input_images; i++) {
			MatrixXf buf_y = MatrixXf::Zero(mem.image_height, mem.image_width);
			SetImageToMatrix(buf_y, mem.degraded_image_name[i], mem, mem.image_height, mem.image_width);
			MatrixXf pad_input = PaddingMartix(buf_y, mem.estimated_kernel_half_height);
			SaveImage(buf_y, "in" + to_string(i), mem);
			mem.input_image_mats.push_back(pad_input);
			SaveImage(mem.input_image_mats[i], "in_pad" + to_string(i) , mem);
		}

		mem.pad_image_height = mem.image_height + mem.estimated_kernel_half_height * 2;
		mem.pad_image_width = mem.image_width + mem.estimated_kernel_half_height * 2;
		mem.pad_image_size = mem.pad_image_height * mem.pad_image_width;

	}


	VectorXf connected_estimated_image_vec = VectorXf::Zero(mem.pad_image_size * mem.num_of_input_images);
	for (int i = 0; i < mem.num_of_input_images; i++) {
		VectorXf blurred_image_vec = MatrixToVector(mem.input_image_mats[i]);
		connected_estimated_image_vec.block(abs(i) * mem.pad_image_size, 0, mem.pad_image_size, 1) = 255 * VectorXf::Ones(mem.pad_image_size) - blurred_image_vec;
		mem.estimated_image_vecs.push_back(blurred_image_vec);
	}

	mem.X = connected_estimated_image_vec * ((mem.estimated_kernel_vecs[0]).transpose());


}