#include "mytool.h"
#include <iostream>
#include <fstream>
#include <time.h>
#include <direct.h>
#include <string>
#include "Member.h"

#include "matrix_tool.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sys/stat.h>



using namespace std;
using namespace Eigen;


void MakeDirectory(Member &mem) {


	mem.folder_name = "out/" + mem.add_title + "/";

	const char* folder2 = mem.folder_name.c_str();

	struct stat st;

	int ret = stat(folder2, &st);

	if (ret == 0) {
		return;
	}

	if (_mkdir(folder2) == 0) {
		cout << "mkdir" << endl;
	}
	else {
		cout << "mkdir error" << endl;
		exit(1);
	}
}

bool copy(const char* src, const char* dst)
{

	std::ifstream ifs(src);
	if (!ifs) {
		std::cerr << "file open error: " << src << '\n';
		return false;
	}

	std::ofstream ofs(dst);
	if (!ofs) {
		std::cerr << "dst open error: " << dst << '\n';
		return false;
	}


	ofs << ifs.rdbuf() << std::flush;


	if (!ifs) {
		std::cerr << "I/O error: " << src << '\n';
		return false;
	}
	if (!ofs) {
		std::cerr << "I/O error: " << dst << '\n';
		return false;
	}

	return true;
}

void GetAllPSNR_SSIM(std::vector<float>& xpsnr, std::vector<float>& xssim, const Member mem) {

	std::vector<VectorXf>  crop_estimated_vecs;
	int pad_l = mem.estimated_kernel_half_height;

	for (int i = 0; i < mem.num_of_input_images; i++) {
		MatrixXf buf = VectortoMatrix(mem.estimated_image_vecs[i], mem.pad_image_height, mem.pad_image_width);
		MatrixXf crop_esti = buf.block(pad_l, pad_l, mem.image_height, mem.image_width);
		MatrixXf crop_true = VectortoMatrix(mem.correct_image_vecs[i], mem.image_height, mem.image_width);
		xssim[i] = CalculateSSIM(crop_true, crop_esti);
		crop_estimated_vecs.push_back(MatrixToVector(crop_esti));
		xpsnr[i] = CalculatePSNR(mem.correct_image_vecs[i], crop_estimated_vecs[i], 255);
	}



}

void WriteLog(const std::vector<float>& xpsnr, const std::vector<float>& xssim, const  Member mem) {
	string log_text_namt = mem.folder_name;
	log_text_namt += "//log_psnr.txt";

	ofstream fs1;
	if (mem.iter == 0) {
		fs1.open(log_text_namt, ios::out);
		if (fs1.fail())
			cout << "can't read txt" << endl;
		fs1 << "iter\ttime\t\tfront\t\tback" << endl;

	}
	else {
		fs1.open(log_text_namt, ios::app);
		if (fs1.fail())
			cout << "can't read txt" << endl;
	}
	fs1 << std::fixed;
	fs1 << std::setw(3) << mem.iter << "\t\t" << std::setprecision(4) << mem.time;
	fs1 << "\t\t" << std::setprecision(4) << xpsnr[0];
	fs1 << "\t\t" << xpsnr[1] << endl;
	fs1.close();



	log_text_namt = mem.folder_name;
	log_text_namt += "//log_ssim.txt";

	if (mem.iter == 0) {
		fs1.open(log_text_namt, ios::out);
		if (fs1.fail())
			cout << "can't read txt" << endl;
		fs1 << "iter\ttime\t\tfront\t\tback" << endl;

	}
	else {
		fs1.open(log_text_namt, ios::app);
		if (fs1.fail())
			cout << "can't read txt" << endl;
	}
	fs1 << std::fixed;
	fs1 << std::setw(3) << mem.iter << "\t\t" << std::setprecision(4) << mem.time;
	fs1 << "\t\t" << std::setprecision(4) << xssim[0];
	fs1 << "\t\t" << xssim[1] << endl;
	fs1.close();


}

void SaveEvalValue(Member mem) {
	std::vector<float> xpsnrs(mem.num_of_input_images);
	std::vector<float> xssims(mem.num_of_input_images);
	GetAllPSNR_SSIM(xpsnrs, xssims, mem);
	cout << "iter:" << mem.iter << endl;
	for (int l = 0; l < mem.num_of_input_images; l++) {
		cout << "xpsnrs:" << xpsnrs[l] << endl;
	}
	for (int l = 0; l < mem.num_of_input_images; l++) {
		cout << "xssim:" << xssims[l] << endl;
	}


	WriteLog(xpsnrs, xssims,  mem);
}

void SaveImg(Member mem) {

	int pad_l = mem.estimated_kernel_half_height;

	string out_name = "//" + to_string(mem.iter) + "_0";
	MatrixXf buf_mat = VectortoMatrix(mem.estimated_image_vecs[0], mem.pad_image_height, mem.pad_image_width);
	SaveImage(buf_mat.block(pad_l, pad_l, mem.image_height, mem.image_width), out_name, mem);

	out_name = "//" + to_string(mem.iter) + "_1";
	buf_mat = VectortoMatrix(mem.estimated_image_vecs[1], mem.pad_image_height, mem.pad_image_width);
	SaveImage(buf_mat.block(pad_l, pad_l, mem.image_height, mem.image_width), out_name, mem);

}