#include "mytool.h"
#include "matrix_tool.h"
#include "matrix_tool.h"
#include "Eigen/Core"
#include "Eigen/Dense"

#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>


using namespace std;
using namespace Eigen;

void MoveGradient(Member& mem);
void calc_data_diffX(const Member mem, MatrixXf& Ediff);
void calc_TV_diffX(const Member mem, MatrixXf& diffTVX);
void RankUpdate(Member& mem);


void Min_rank_method(Member mem) {


	mem.start = clock();

	for (mem.iter = 1; mem.iter <= mem.num_of_iteration + 1; ++mem.iter) {

		cout << "iter:" << mem.iter << endl;

		MoveGradient(mem);

		RankUpdate(mem);


		if (mem.iter % mem.log_save_iteration == 0 && mem.mode == "simulation") {
			SaveEvalValue(mem);
		}

		if (mem.iter % mem.img_save_iteration == 0) {
			SaveImg(mem);
		}

	}

}

void MoveGradient(Member& mem) {

	MatrixXf data_diff;
	MatrixXf TV_diff;


	data_diff = MatrixXf::Zero(mem.num_of_input_images * mem.pad_image_size, mem.estimated_kernel_size + 1);
	TV_diff = MatrixXf::Zero(mem.num_of_input_images * mem.pad_image_size, mem.estimated_kernel_size + 1);

	calc_data_diffX(mem, data_diff);

	if (mem.smoothing_term_param != 0) {
		calc_TV_diffX(mem, TV_diff);
	}


	MatrixXf dX;

	dX = data_diff + mem.smoothing_term_param * TV_diff;

	static float t;
	static float next_t;
	static MatrixXf prev_X;

	if (mem.prox_mode == "APG") {
		if (mem.iter == 1) {
			t = 1;
			prev_X = mem.X;
		}

		next_t = (1 + sqrt(1 + 4 * t * t)) / 2;
		MatrixXf W = mem.X + (mem.X - prev_X) * (t - 1) / next_t;
		prev_X = mem.X;
		t = next_t;
		mem.X = W - mem.step_size * dX;
	}
	else if ( mem.prox_mode == "PG") {

		mem.X = mem.X - mem.step_size * dX;
	}
	else {
		cout << mem.prox_mode << "is nonexistent mode" << endl;
		return;
	}
	if (mem.enable_u8_clamp) {
		for (int j = 0; j < mem.X.cols(); j++) {
			for (int i = 0; i < mem.X.rows(); i++) {
				if (mem.X(i, j) < 0) mem.X(i, j) = 0;
				else if (mem.X(i, j) > 255) mem.X(i, j) = 255;
			}
		}
	}

	VectorXf prev_estimated_image_vecs0, prev_estimated_image_vecs1;

	prev_estimated_image_vecs0 = mem.estimated_image_vecs[0];
	prev_estimated_image_vecs1 = mem.estimated_image_vecs[1];

	mem.end = clock();
	mem.time = (double)(mem.end - mem.start) / CLOCKS_PER_SEC;

	for (int im_num = 0; im_num < 2; im_num++) {
		VectorXf estimated_front_vec = -(mem.X.block(im_num * mem.pad_image_size, mem.estimated_kernel_size, mem.pad_image_size, 1) - 255 * VectorXf::Ones(mem.pad_image_size));
		mem.estimated_image_vecs[im_num] = estimated_front_vec;
	}

}

void calc_data_diffX(const Member mem, MatrixXf& Ediff) {


	const int etimated_kernel_half_width = mem.estimated_kernel_half_width;
	const int etimated_kernel_half_height = mem.estimated_kernel_half_height;
	const int image_i_end = mem.pad_image_height - etimated_kernel_half_height;
	const int image_j_end = mem.pad_image_width - etimated_kernel_half_width;


	for (int l = 0; l < mem.num_of_input_images; l++) {

		MatrixXf estimated_blurred_image_mat = MatrixXf::Zero(mem.pad_image_height, mem.pad_image_width);
		for (int i = etimated_kernel_half_height; i < image_i_end; i++) {
			for (int j = etimated_kernel_half_width; j < image_j_end; j++) {

				for (int h = 0; h < mem.estimated_kernel_height; h++) {
					for (int k = 0; k < mem.estimated_kernel_width; k++) {
						estimated_blurred_image_mat(i, j) += mem.X((i - etimated_kernel_half_height + h) * mem.pad_image_width + (j - etimated_kernel_half_width + k) + l * mem.pad_image_size, h * mem.estimated_kernel_width + k);
					}
				}
			}
		}


		MatrixXf  error_mat = MatrixXf::Zero(mem.pad_image_height, mem.pad_image_width);
		int im_num = abs(l - 1);

		estimated_blurred_image_mat = horizontal_flip(estimated_blurred_image_mat);

		VectorXf estimated_front_vec = -(mem.X.block(im_num * mem.pad_image_size, mem.estimated_kernel_size, mem.pad_image_size, 1) - 255 * VectorXf::Ones(mem.pad_image_size));
		MatrixXf estimated_front_mat = VectortoMatrix(estimated_front_vec, mem.pad_image_height, mem.pad_image_width);

		for (int i = etimated_kernel_half_height; i < image_i_end; i++) {
			for (int j = etimated_kernel_half_width; j < image_j_end; j++) {

				float estimated_val = -estimated_blurred_image_mat(i, j) + estimated_front_mat(i, j);
				error_mat(i, j) = estimated_val - mem.input_image_mats[im_num](i, j);


			}
		}

		MatrixXf  buf = error_mat;
		error_mat *= 0;
		error_mat.block(etimated_kernel_half_height, etimated_kernel_half_width, mem.pad_image_height - 2 * etimated_kernel_half_height, mem.pad_image_width - 2 * etimated_kernel_half_width) = buf.block(etimated_kernel_half_height, etimated_kernel_half_width, mem.pad_image_height - 2 * etimated_kernel_half_height, mem.pad_image_width - 2 * etimated_kernel_half_width);
		MatrixXf flip_error_mat = horizontal_flip(error_mat);

		for (int i = etimated_kernel_half_height; i < image_i_end; i++) {
			for (int j = etimated_kernel_half_width; j < image_j_end; j++) {

				for (int h = 0; h < mem.estimated_kernel_height; h++) {
					for (int k = 0; k < mem.estimated_kernel_width; k++) {
						if (j - etimated_kernel_half_width + k >= 0 && i - etimated_kernel_half_height + h >= 0 && i - etimated_kernel_half_height + h < mem.pad_image_height && j - etimated_kernel_half_width + k < mem.pad_image_width) {
							Ediff((i - etimated_kernel_half_height + h) * mem.pad_image_width + (j - etimated_kernel_half_width + k) + l * mem.pad_image_size, h * mem.estimated_kernel_width + k) = -2 * 0.5 * flip_error_mat(i, j);
						}
					}
				}
				Ediff(im_num * mem.pad_image_size + i * mem.pad_image_width + j, mem.estimated_kernel_size) = -2 * 0.5 * error_mat(i, j);
			}

		}


	}

}

void calc_TV_diffX(const Member mem, MatrixXf& diffTVX) {


	float epsilion = 1;


	for (int i = 0; i < mem.pad_image_size; ++i) {

		float buf = 0;

		if (i % (mem.pad_image_width) < mem.pad_image_width - 1 && int(i / mem.pad_image_width) < mem.pad_image_height - 1 && i % (mem.pad_image_width) > 0 && int(i / mem.pad_image_width) > 0) {

			float y_h_2_y_v_2 = pow(mem.X(i + 1, mem.estimated_kernel_size) - mem.X(i, mem.estimated_kernel_size), 2) + pow(mem.X(i + mem.pad_image_width, mem.estimated_kernel_size) - mem.X(i, mem.estimated_kernel_size), 2);
			float diff_y_h_2_y_v_2 = mem.X(i, mem.estimated_kernel_size) - mem.X(i + 1, mem.estimated_kernel_size) + mem.X(i, mem.estimated_kernel_size) - mem.X(i + mem.pad_image_width, mem.estimated_kernel_size);

			buf = diff_y_h_2_y_v_2 / sqrt(y_h_2_y_v_2 + epsilion);

			float y_h_1_2_y_v_2 = pow(mem.X(i, mem.estimated_kernel_size) - mem.X(i - 1, mem.estimated_kernel_size), 2) + pow(mem.X(i + mem.pad_image_width - 1, mem.estimated_kernel_size) - mem.X(i - 1, mem.estimated_kernel_size), 2);
			float diff_y_h_1_y_v_2 = mem.X(i, mem.estimated_kernel_size) - mem.X(i - 1, mem.estimated_kernel_size);

			buf += diff_y_h_1_y_v_2 / sqrt(y_h_1_2_y_v_2 + epsilion);

			float y_h_2_y__v_1_2 = pow(mem.X(i, mem.estimated_kernel_size) - mem.X(i - mem.pad_image_width, mem.estimated_kernel_size), 2) + pow(mem.X(i - mem.pad_image_width + 1, mem.estimated_kernel_size) - mem.X(i - mem.pad_image_width, mem.estimated_kernel_size), 2);
			float diff_y_h_2_y__v_1_2 = mem.X(i, mem.estimated_kernel_size) - mem.X(i - mem.pad_image_width, mem.estimated_kernel_size);

			buf += diff_y_h_2_y__v_1_2 / sqrt(y_h_2_y__v_1_2 + epsilion);

		}
		diffTVX(i, mem.estimated_kernel_size) = buf;

	}

	for (int i = mem.pad_image_size; i < mem.pad_image_size * 2; ++i) {

		float buf = 0;



		if ((i - mem.pad_image_size) % (mem.pad_image_width) < mem.pad_image_width - 1 && int((i - mem.pad_image_size) / mem.pad_image_width) < mem.pad_image_height - 1 && i % (mem.pad_image_width) > 0 && int((i - mem.pad_image_size) / mem.pad_image_width) > 0) {

			float y_h_2_y_v_2 = pow(mem.X(i + 1, mem.estimated_kernel_size) - mem.X(i, mem.estimated_kernel_size), 2) + pow(mem.X(i + mem.pad_image_width, mem.estimated_kernel_size) - mem.X(i, mem.estimated_kernel_size), 2);
			float diff_y_h_2_y_v_2 = mem.X(i, mem.estimated_kernel_size) - mem.X(i + 1, mem.estimated_kernel_size) + mem.X(i, mem.estimated_kernel_size) - mem.X(i + mem.pad_image_width, mem.estimated_kernel_size);

			buf = diff_y_h_2_y_v_2 / sqrt(y_h_2_y_v_2 + epsilion);

			float y_h_1_2_y_v_2 = pow(mem.X(i, mem.estimated_kernel_size) - mem.X(i - 1, mem.estimated_kernel_size), 2) + pow(mem.X(i + mem.pad_image_width - 1, mem.estimated_kernel_size) - mem.X(i - 1, mem.estimated_kernel_size), 2);
			float diff_y_h_1_y_v_2 = mem.X(i, mem.estimated_kernel_size) - mem.X(i - 1, mem.estimated_kernel_size);

			buf += diff_y_h_1_y_v_2 / sqrt(y_h_1_2_y_v_2 + epsilion);

			float y_h_2_y__v_1_2 = pow(mem.X(i, mem.estimated_kernel_size) - mem.X(i - mem.pad_image_width, mem.estimated_kernel_size), 2) + pow(mem.X(i - mem.pad_image_width + 1, mem.estimated_kernel_size) - mem.X(i - mem.pad_image_width, mem.estimated_kernel_size), 2);
			float diff_y_h_2_y__v_1_2 = mem.X(i, mem.estimated_kernel_size) - mem.X(i - mem.pad_image_width, mem.estimated_kernel_size);

			buf += diff_y_h_2_y__v_1_2 / sqrt(y_h_2_y__v_1_2 + epsilion);

		}
		diffTVX(i, mem.estimated_kernel_size) = buf;

	}

}

void RankUpdate(Member& mem) {


	MatrixXf XTX = (mem.X).transpose() * mem.X;
	VectorXf V0, U0;

	int v_colo_num = mem.estimated_kernel_size * 2 + 1;



	if (mem.threshold_method == "hard" || mem.threshold_method == "SVP") {
		MatrixXf D, V;
		SelfAdjointEigenSolver<MatrixXf> eig(XTX);


		D = eig.eigenvalues();
		V = eig.eigenvectors();

		float max = D.maxCoeff();


		for (int i = 0; i < v_colo_num; i++) {
			if (D(i) == max) {
				V0 = V.col(i);
				break;
			}
		}

		U0 = mem.X * V0 / sqrt(max);
		V0 = V0 * sqrt(max);
		mem.X = U0 * V0.transpose();
	}
	else if (mem.threshold_method == "soft" || mem.threshold_method == "ST") {
		JacobiSVD<MatrixXf> SVD(mem.X, ComputeThinU | ComputeThinV);
		VectorXf vec_s = SVD.singularValues();

		cout << vec_s << endl << endl;

		const float thresh = mem.soft_thresh_val;
		for (int i = 0; i < vec_s.size(); i++) {

			if (thresh <= vec_s(i)) {
				vec_s(i) = vec_s(i) - thresh;
			}
			else {
				vec_s(i) = 0;
			}

		}
		cout << vec_s << endl << endl;
		mem.X = SVD.matrixU() * vec_s.asDiagonal() * SVD.matrixV().transpose();

		U0 = SVD.matrixU().col(0);
		V0 = SVD.matrixV().col(0) * vec_s(0);
	}
	else {

		cout << "no threshold mode" << endl;
		exit(1);
	}

	float  delta_1 = 0;


	VectorXf v_vec = V0.block(0, 0, mem.estimated_kernel_size, 1);
	delta_1 = V0[mem.estimated_kernel_size];

	mem.estimated_kernel_vecs[0] = (1 / delta_1) * V0.block(0, 0, mem.estimated_kernel_size, 1);


}
