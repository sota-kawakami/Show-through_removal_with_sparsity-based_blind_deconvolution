#include "Member.h"
#include "mytool.h"

#include "initialize.h"
#include "minrank_method.h"
#include "sharma_method.h"
#include "nmf_method.h"

#include <iostream>


using namespace std;


int main(int argc, char* argv[])
{
	cout << "read:" << argv[1] << endl;
	if (argc < 1) {
		cout << "must input json" << endl;
		return 0;
	}

	Member my_mem(argv[1]);

	cout << "make dir" << endl;
	MakeDirectory(my_mem);
	copy(argv[1], (my_mem.folder_name + "/in.json").c_str());


	if (my_mem.mode == "simulation") {
		MakeKernelAndBlur(my_mem);
	}

	Initialize(my_mem);

	if (my_mem.mode == "simulation" && my_mem.method != "nmf") {
		SaveEvalValue(my_mem);
	}
	SaveImg(my_mem);



	if (my_mem.method == "minrank") {
		Min_rank_method(my_mem);
	}
	else if (my_mem.method == "sharma") {
		Sharma_method(my_mem);

	}
	else if (my_mem.method == "nmf") {
		NMF_method(my_mem);
	}
	else {
		cout << "The only methods are minrank or sharma or nmf only." << endl;
	}

	return 0;

}

