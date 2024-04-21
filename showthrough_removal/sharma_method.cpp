#include "sharma_method.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Member.h"
#include "matrix_tool.h"
#include "mytool.h"
#include "initialize.h"
#include <vector>


using namespace Eigen;
using namespace std;


MatrixXf printCheck(const VectorXf in_vec,const VectorXf in_vec2 ,Member mem, const  int back1, const  int back2) {


	MatrixXf flag_mat = MatrixXf::Zero(mem.pad_image_height, mem.pad_image_width);

	const int M = 1;

	MatrixXf dai_x_mat = 255 * MatrixXf::Ones(2*M + mem.pad_image_height, 2*M + mem.pad_image_width);
	MatrixXf dai_x_mat2 = 255 * MatrixXf::Ones(2 * M + mem.pad_image_height, 2 * M + mem.pad_image_width);
	MatrixXf x_mat = VectortoMatrix(in_vec, mem.pad_image_height, mem.pad_image_width);
	MatrixXf x_mat2 = VectortoMatrix(in_vec2, mem.pad_image_height, mem.pad_image_width);

	dai_x_mat.block(M, M, mem.pad_image_height, mem.pad_image_width) = x_mat;
	dai_x_mat2.block(M, M, mem.pad_image_height, mem.pad_image_width) = x_mat2;

	const float thresh = mem.threshold;

	for (int i = 0; i < mem.pad_image_height; i++) {
		for (int j = 0; j < mem.pad_image_width; j++) {

			MatrixXf neighborhood_mat = dai_x_mat.block( i, j  ,2*M + 1 , 2*M + 1);
			MatrixXf neighborhood_mat2 = dai_x_mat2.block(i, j, 2 * M + 1, 2 * M + 1);

			if (neighborhood_mat.minCoeff() > back1 * thresh && neighborhood_mat2.minCoeff() <= thresh * back2){
				flag_mat(i, j) = 1;
	
			}


		}
	}


	return flag_mat;

}

void Sharma_method(Member mem) {



	VectorXf in_image_vec1_1 = MatrixToVector(mem.input_image_mats[0]);
	VectorXf in_image_vec1_2 = MatrixToVector( horizontal_flip(mem.input_image_mats[1]));

	VectorXf in_image_vec2_1 = MatrixToVector(mem.input_image_mats[1]);
	VectorXf in_image_vec2_2 = MatrixToVector(horizontal_flip(mem.input_image_mats[0]));

	const int back1 = mem.back_pixel_val;
	const int back2 = back1;

	

	VectorXf df_vec1 = VectorXf::Zero(mem.pad_image_size);
	VectorXf ab_vec1 = VectorXf::Zero(mem.pad_image_size);
	VectorXf df_vec2 = VectorXf::Zero(mem.pad_image_size);
	VectorXf ab_vec2 = VectorXf::Zero(mem.pad_image_size);

	for (int i = 0; i < mem.pad_image_size; i++) {
		df_vec1[i] = -log(in_image_vec1_1[i] / back1);
		ab_vec1[i] = 1 - (in_image_vec1_2[i] / back2);
		df_vec2[i] = -log(in_image_vec2_1[i] / back2);
		ab_vec2[i] = 1 - (in_image_vec2_2[i] / back1);
	}

	const VectorXf df_vec_s1 = df_vec1;
	const VectorXf df_vec_s2 = df_vec2;

	const int M = mem.kernel_l;
	MatrixXf w_mats1 = MatrixXf::Zero(M, M);
	MatrixXf w_mats2 = MatrixXf::Zero(M, M);


	MatrixXf flag_mat1 = printCheck(in_image_vec1_1, in_image_vec1_2, mem, back1, back2);
	MatrixXf flag_mat2 = printCheck(in_image_vec2_1, in_image_vec2_2, mem, back2, back1);

	const float step_size = mem.step_size;
	
	mem.start = clock();

	for (mem.iter = 1; mem.iter <= mem.num_of_iteration; ++mem.iter) {

		cout << "iter:" << mem.iter << endl;

		for (int i = 0; i < mem.pad_image_height; i++) {
			for (int j = 0; j < mem.pad_image_width; j++) {

				float buf1 = 0;
				float buf2 = 0;

				for (int m = 0; m < M; m++) {
					for (int n = 0; n < M; n++) {
						int m2 = m - int(M / 2);
						int n2 = n - int(M / 2);
					
						if((i + m2) * mem.pad_image_width + (j + n2) >= 0 && (i + m2) * mem.pad_image_width + (j + n2) < ab_vec1.size())
							buf1 += w_mats1(m, n) * ab_vec1((i + m2) * mem.pad_image_width + (j + n2));

						if ((i + m2) * mem.pad_image_width + (j + n2) >= 0 && (i + m2) * mem.pad_image_width + (j + n2) < ab_vec2.size())
							buf2 += w_mats2(m, n) * ab_vec2((i + m2) * mem.pad_image_width + (j + n2));

					}
				}
				df_vec1[i * mem.pad_image_width + j] = df_vec_s1[i * mem.pad_image_width + j] - buf1;
				df_vec2[i * mem.pad_image_width + j] = df_vec_s2[i * mem.pad_image_width + j] - buf2;

				if (flag_mat1(i, j) != 0) {

					for (int m = 0; m < M; m++) {
						for (int n = 0; n < M; n++) {
							int m2 = m - int(M / 2);
							int n2 = n - int(M / 2);
							if ((i + m2) * mem.pad_image_width + (j + n2) >= 0 && (i + m2) * mem.pad_image_width + (j + n2) < ab_vec1.size())
								w_mats1(m, n) = w_mats1(m, n) + step_size * ab_vec1((i + m2) * mem.pad_image_width + (j + n2)) * df_vec1[i * mem.pad_image_width + j];
							if (w_mats1(m, n) < 0) w_mats1(m, n) = 0;

						}
					}

				}
				if (flag_mat2(i, j) != 0) {
					for (int m = 0; m < M; m++) {
						for (int n = 0; n < M; n++) {
							int m2 = m - int(M / 2);
							int n2 = n - int(M / 2);
							if ((i + m2) * mem.pad_image_width + (j + n2) >= 0 && (i + m2) * mem.pad_image_width + (j + n2) < ab_vec2.size())
								w_mats2(m, n) = w_mats2(m, n) + step_size * ab_vec2((i + m2) * mem.pad_image_width + (j + n2)) * df_vec2[i * mem.pad_image_width + j];
							if (w_mats2(m, n) < 0) w_mats2(m, n) = 0;

						}
					}

				}

			}

		}

		
		for (int i = 0; i < mem.pad_image_size; i++) {
			mem.estimated_image_vecs[0][i] = back1 * exp(-df_vec1[i]);
			mem.estimated_image_vecs[1][i] = back2 * exp(-df_vec2[i]);		
		}

		mem.end = clock();
		mem.time = (double)(mem.end - mem.start) / CLOCKS_PER_SEC;


		if (mem.iter % mem.log_save_iteration == 0 && mem.mode == "simulation") {
			SaveEvalValue(mem);
		}

		if (mem.iter % mem.img_save_iteration == 0) {
			SaveImg(mem);
		}



	}


}