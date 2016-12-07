#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <float.h>

using namespace std;

#ifdef WIN32
const string Dir = "\\"; // for Windows
#else
const string Dir = "/"; // for Mac OS or Linux
#endif

const int Num_ds = 4;
const int Num_y = 7;

class Part_2
{
private:
	const string dataset[Num_ds] = { "EN", "ES", "CN", "SG" };
	const wstring index_to_y[Num_y] = {
		L"O", 
		L"B-positive", L"I-positive",
		L"B-neutral", L"I-neutral",
		L"B-negative", L"I-negative"
	}; // the array of states
	map <wstring, int> y_to_index; // map the states to indices

	// initialize the y_to_index array
	void pre() {
		for (int i = 0; i < Num_y; ++i) {
			y_to_index[index_to_y[i]] = i;
		}
	}

	// the state structure
	struct y_x {
		int y;
		wstring x;

		y_x() {
		}
		y_x(int _y, const wstring& _x) : y(_y), x(_x) {
		}

		bool operator < (const y_x& r) const {
			if (y != r.y) {
				return y < r.y;
			}
			return x < r.x;
		}
	};

	map <y_x, int> count_xy;
	int count_y[Num_y];
	set <wstring> appear; // contains the word that appears

	// the MLE of emission parameters
	double emission_MLE(wstring x, int y) {
		return (double)count_xy[y_x(y, x)] / count_y[y];
	}

	// the MLE of emission parameters when a fixed probability is assigned to all such new words
	double emission_fixed(wstring x, int y) {
		if (appear.find(x) == appear.end()) {
			return 1.0 / (count_y[y] + 1);
		}
		return (double)count_xy[y_x(y, x)] / (count_y[y] + 1);
	}

	// train the HMM model using MLE
	void train(int which) {
		string input = dataset[which] + Dir + "train";
		wifstream wfin(input.c_str());
		if (!wfin) {
			cerr << "E0: File not found!" << endl;
			return;
		}
		memset(count_y, 0, sizeof count_y);
		count_xy.clear();
		appear.clear();
		wstring x, y;
		while (wfin >> x >> y) {
			int idx = y_to_index[y];
			count_xy[y_x(idx, x)]++;
			count_y[idx]++;
			appear.insert(x);
		}
		wfin.close();
	}

	// test the result using viterbi algorithm	
	void test(int which) {
		string input = dataset[which] + Dir + "dev.in";
		string output = dataset[which] + Dir + "dev.p2.out";
		wifstream wfin(input.c_str());
		wofstream wfout(output.c_str());
		if (!wfin) {
			cerr << "E1: File not found!" << endl;
			return;
		}
		wstring x;
		while (getline(wfin, x)) {
			if (x.empty()) {
				wfout << endl;
				continue;
			}
			int y_star = -1;
			double score = 0;
			for (int i = 0; i < Num_y; ++i) {
				double cur = emission_fixed(x, i);
				if (cur > score) {
					score = cur;
					y_star = i;
				}
			}
			wfout << x << ' ' << index_to_y[y_star] << endl;
		}
		wfin.close();
	}

public:
	Part_2() {
		pre();
	}

	void work() {
		for (int i = 0; i < Num_ds; ++i) {
			train(i);
			test(i);
			clog << dataset[i] << " finished." << endl;
		}
	}
};

int main()
{
	Part_2 p2;
	p2.work();
	return 0;
}
