#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <float.h>

using namespace std;

#ifdef WIN32
const string Dir = "\\"; // for Windows
#else
const string Dir = "/"; // for Mac OS or Linux
#endif

const int Num_ds = 4;
const int Num_y = 7;

class Part_3
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
	set <wstring> appear;  // contains the word that appears
	int count_yy[Num_y][Num_y];
	int count_start_y[Num_y], count_y_stop[Num_y];
	int count_start;

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

	// the MLE of transition parameters
	double transition_MLE(int u, int v) { // -1: START; -2: STOP
		if (u == -1) {
			return (double)count_start_y[v] / count_start;
		}
		if (v == -2) {
			return (double)count_y_stop[u] / count_y[u];
		}
		return (double)count_yy[u][v] / count_y[u];
	}

	// the MLE of transition parameters when a fixed probability is assigned to all such new words
	double transition_fixed(int u, int v) { // -1: START; -2: STOP
		if (u == -1) {
			if (count_start_y[v] == 0) {
				return 1.0 / (count_start + 1);
			}
			return (double)count_start_y[v] / (count_start + 1);
		}
		if (v == -2) {
			if (count_y_stop[u] == 0) {
				return 1.0 / (count_y[u] + 1);
			}
			return (double)count_y_stop[u] / (count_y[u] + 1);
		}
		if (count_yy[u][v] == 0) {
			return 1.0 / (count_y[u] + 1);
		}
		return (double)count_yy[u][v] / (count_y[u] + 1);
	}

	// the viterbi algorithm to predicate the results 
	void viterbi(const vector<wstring> &adj, vector<int> &res) {
		int n = (int)adj.size();
		double pu[Num_y], pv[Num_y];
		vector <int> fa[Num_y];
		for (int i = 0; i < Num_y; ++i) {
			pv[i] = log(transition_MLE(-1, i) * emission_fixed(adj[0], i));
		}
		// compute the best score of each node in each layer using Dynamic Programming algorithm
		for (int i = 1; i < n; ++i) {
			memcpy(pu, pv, sizeof pu);
			for (int j = 0; j < Num_y; ++j) {
				double score = -DBL_MAX;
				int p = -1;
				double emission_value = log(emission_fixed(adj[i], j));
				for (int k = 0; k < Num_y; ++k) {
					double cur = pu[k] + log(transition_MLE(k, j));
					if (cur > score || p == -1) {
						score = cur;
						p = k;
					}
				}
				if (p == -1) {
					cout << "E2: viterbi error!" << endl;
					return;
				}
				pv[j] = score + emission_value;
				fa[j].push_back(p);
			}
		}
		double score_n = -DBL_MAX;
		int fa_n = -1;
		// get the path using the record of moving forward
		for (int i = 0; i < Num_y; ++i) {
			double cur = pv[i] + log(transition_MLE(i, -2));
			if (cur > score_n || fa_n == -1) {
				score_n = cur;
				fa_n = i;
			}
		}
		res.push_back(fa_n);
		int behind = fa_n;
		for (int i = n - 2; i >= 0; --i) {
			res.push_back(fa[behind][i]);
			behind = fa[behind][i];
		}
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
		memset(count_yy, 0, sizeof count_yy);
		memset(count_start_y, 0, sizeof count_start_y);
		memset(count_y_stop, 0, sizeof count_y_stop);
		count_start = 0;
		count_xy.clear();
		appear.clear();
		wstring ws;
		bool start_of_sentence = true;
		int pre_y = -1;
		while (getline(wfin, ws)) {
			if (ws.empty()) {
				count_y_stop[pre_y]++;
				start_of_sentence = true;
				pre_y = -1;
				continue;
			}
			wstringstream wss(ws);
			wstring x, y;
			wss >> x >> y;
			int idx = y_to_index[y];
			count_xy[y_x(idx, x)]++;
			count_y[idx]++;
			appear.insert(x);
			if (start_of_sentence) {
				count_start++;
				count_start_y[idx]++;
				start_of_sentence = false;
				pre_y = idx;
				continue;
			}
			count_yy[pre_y][idx]++;
			pre_y = idx;
		}
		wfin.close();
	}

	// test the result using viterbi algorithm
	void test(int which) {
		string input = dataset[which] + Dir + "dev.in";
		string output = dataset[which] + Dir + "dev.p3.out";
		wifstream wfin(input.c_str());
		wofstream wfout(output.c_str());
		if (!wfin) {
			cerr << "E1: File not found!" << endl;
			return;
		}
		wstring x;
		while (getline(wfin, x)) {
			vector <wstring> adj;
			adj.push_back(x);
			while (getline(wfin, x)) {
				if (x.empty()) {
					break;
				}
				adj.push_back(x);
			}
			vector <int> res;
			viterbi(adj, res);
			int n = (int)adj.size();
			for (int i = 0; i < n; ++i) {
				wfout << adj[i] << ' ' << index_to_y[res[n - i - 1]] << endl;
			}
			wfout << endl;
		}
		wfin.close();
		wfout.close();
	}

public:
	Part_3() {
		pre();
	}

	void work() {
		for (int i = 0; i < Num_ds; ++i) {
			train(i);
			clog << dataset[i] << " train finished." << endl;
			test(i);
			clog << dataset[i] << " test finished." << endl;
		}
	}
};

int main()
{
	Part_3 p3;
	p3.work();
	return 0;
}
