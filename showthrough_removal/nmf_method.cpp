
#include "nmf_method.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Member.h"
#include "matrix_tool.h"
#include "mytool.h"
#include "initialize.h"
#include <vector>



using namespace Eigen;
using namespace std;



void NMF_method(Member mem) {


	VectorXf y_vec_1 = MatrixToVector(mem.input_image_mats[0]);
	VectorXf y_vec_2 = MatrixToVector(horizontal_flip(mem.input_image_mats[1]));

	MatrixXf A = MatrixXf::Random(2, 2)*255;
	MatrixXf X = MatrixXf::Random(2, mem.pad_image_size);

	MatrixXf Y = MatrixXf::Zero(2, mem.pad_image_size);
	Y.block(0, 0, 1, mem.pad_image_size) = y_vec_1.transpose();
	Y.block(1, 0, 1, mem.pad_image_size) = y_vec_2.transpose();


	srand((unsigned int)time(0));
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			if (A(i, j) < 0) {
				A(i, j) = -A(i, j);
			}
		}
	}

	for (int i = 0; i < mem.image_size; i++) {
		for (int j = 0; j < 2; j++) {
			if (X(j, i) < 0) {
				X(j, i) = -X(j, i);
			}
		}
	}

	float step_size = mem.step_size;

	mem.start = clock();

	if ( mem.mode == "simulation") {
		VectorXf x2 = X.block(1, 0, 1, mem.pad_image_size).transpose();
		MatrixXf x2_mat = horizontal_flip(VectortoMatrix(x2, mem.pad_image_height, mem.pad_image_width));


		mem.estimated_image_vecs[0] = X.block(0, 0, 1, mem.pad_image_size).transpose();
		mem.estimated_image_vecs[1] = MatrixToVector(x2_mat);
		float gain1 = mem.back_pixel_val / mem.estimated_image_vecs[0](0);
		float gain2 = mem.back_pixel_val / mem.estimated_image_vecs[1](0);

		mem.estimated_image_vecs[0] *= gain1;
		mem.estimated_image_vecs[1] *= gain2;
		SaveEvalValue(mem);
	}

	for (mem.iter = 1; mem.iter <= mem.num_of_iteration; ++mem.iter) {

		cout << "iter:" <<  mem.iter << endl;

		MatrixXf V = MatrixXf::Zero(2, mem.pad_image_size);
		float w_param = 6 * pow(10, 5);
		for (int i = 0; i < mem.pad_image_size; i++) {
			V(0,i) = exp( - ( pow(X(0,i), 2) + pow(X(1, i), 2))   / (2* pow(22, 2)) ) * w_param;
			V(1, i) = exp(-(pow(X(0, i), 2) + pow(X(1, i), 2)) / (2 * pow(22, 2))) * w_param;
		}


		MatrixXf dA = (A * X - Y) * (X.transpose());
		MatrixXf dX = (A.transpose()) * (A * X - Y);
		A = A - step_size * dA;
		X = X - step_size * dX - step_size * V*0;


		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				if (A(i, j) < 0) {
					A(i, j) = 0;
				}

			}
		}
		

		for (int i = 0; i < mem.pad_image_size; i++) {
			for (int j = 0; j < 2; j++) {
				if (X(j, i) < 0) {
					X(j, i) = 0;
				}
			}
		}

		VectorXf x2 = X.block(1, 0, 1, mem.pad_image_size).transpose();
		MatrixXf x2_mat = horizontal_flip(VectortoMatrix(x2, mem.pad_image_height, mem.pad_image_width));


		mem.estimated_image_vecs[0] = X.block(0, 0, 1, mem.pad_image_size).transpose();
		mem.estimated_image_vecs[1] = MatrixToVector(x2_mat);
		float gain1 = mem.back_pixel_val / mem.estimated_image_vecs[0](0);
		float gain2 = mem.back_pixel_val / mem.estimated_image_vecs[1](0);

		mem.estimated_image_vecs[0] *= gain1;
		mem.estimated_image_vecs[1] *= gain2;

		mem.end = clock();
		mem.time = (double)(mem.end - mem.start) / CLOCKS_PER_SEC;
		

		if (mem.iter % mem.img_save_iteration == 0) {

			SaveImg(mem);

		}

		if (mem.iter % mem.log_save_iteration == 0 && mem.mode == "simulation") {
			SaveEvalValue(mem);
		}



	}


}