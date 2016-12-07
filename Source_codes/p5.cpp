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
#include <algorithm>
#include <cstring>
#include <float.h>

using namespace std;

#ifdef WIN32
const string Dir = "\\"; // for Windows
#else
const string Dir = "/"; // for Mac OS or Linux
#endif

const int Num_ds = 2;
const int Num_y = 7;
const int Num_k = 10;

class Part_5
{
private:
	const string dataset[Num_ds] = { "EN", "ES" };
	// the array of states
	const wstring index_to_y[Num_y] = {
		L"O",
		L"B-positive", L"I-positive",
		L"B-neutral", L"I-neutral",
		L"B-negative", L"I-negative"
	};
	// the weight for different ranking 
	const double weight[Num_k] = {
		1.0 / 1.0,
		0.9 / 1.5,
		0.8 / 2.0,
		0.7 / 2.5,
		0.6 / 3.0,
		0.5 / 3.5,
		0.4 / 4.0,
		0.3 / 4.5,
		0.2 / 5.0,
		0.1 / 5.5,
	};
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

	// the score structure for each node
	struct score_fa {
		double score;
		int fa, k; // father and father's ranking 

		score_fa(double _score = 0, int _fa = 0, int _k = 0) : score(_score), fa(_fa), k(_k) {
		}

		bool operator < (const score_fa &r) const {
			if (score != r.score) {
				return score > r.score;
			}
			return k < r.k;
		}
	};

	//store the first Num_k results
	struct vec_array {
		vector <int> a[Num_k];
	};

	map <y_x, int> count_xy;
	int count_y[Num_y];
	set <wstring> appear; // contains the word that appears
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
	void viterbi(const vector<wstring> &adj, vec_array &res) {
		int n = (int)adj.size();
		double pu[Num_y][Num_k], pv[Num_y][Num_k];
		vector <int> fa[Num_y][Num_k];
		vector <int> fa_k[Num_y][Num_k];
		for (int i = 0; i < Num_y; ++i) {
			pv[i][0] = log(transition_MLE(-1, i) * emission_fixed(adj[0], i));
			for (int j = 1; j < Num_k; ++j) {
				pv[i][j] = -DBL_MAX;
			}
		}
		// compute first Num_k scores of each node in each layer using Dynamic Programming algorithm
		for (int i = 1; i < n; ++i) {
			memcpy(pu, pv, sizeof pu);
			for (int j = 0; j < Num_y; ++j) {
				double score = -DBL_MAX;
				vector <score_fa> all;
				double emission_value = log(emission_fixed(adj[i], j));
				for (int k = 0; k < Num_y; ++k) {
					for (int p = 0; p < Num_k; ++p) {
						double cur = pu[k][p] + log(transition_MLE(k, j));
						all.push_back(score_fa(cur, k, p));
					}
				}
				sort(all.begin(), all.end());
				for (int p = 0; p < Num_k; ++p) {
					pv[j][p] = all[p].score + emission_value;
					fa[j][p].push_back(all[p].fa);
					fa_k[j][p].push_back(all[p].k);
				}
			}
		}
		vector <score_fa> all_n;
		for (int i = 0; i < Num_y; ++i) {
			for (int p = 0; p < Num_k; ++p) {
				double cur = pv[i][p] + log(transition_MLE(i, -2));
				all_n.push_back(score_fa(cur, i, p));
			}
		}
		sort(all_n.begin(), all_n.end());

		// get the first Num_k scores
		for (int cnt = 0; cnt < Num_k; ++cnt) {
			res.a[cnt].push_back(all_n[cnt].fa);
			int behind = all_n[cnt].fa, behind_k = all_n[cnt].k;
			for (int i = n - 2; i >= 0; --i) {
				res.a[cnt].push_back(fa[behind][behind_k][i]);
				int behind_t = fa[behind][behind_k][i];
				behind_k = fa_k[behind][behind_k][i];
				behind = behind_t;
			}
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
		string output = dataset[which] + Dir + "dev.p5.out";
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
			vec_array res;
			viterbi(adj, res);
			int n = (int)adj.size();
			double vote[Num_y];
			// vote for each transition according to the weight
			for (int i = 0; i < n; ++i) {
				memset(vote, 0, sizeof vote);
				for (int j = 0; j < Num_k; ++j) {
					int now = res.a[j][n - i - 1];
					vote[now] += weight[j];
				}
				double best = -1;s
				int p = -1;
				for (int j = 0; j < Num_y; ++j) {
					if (vote[j] > best) {
						best = vote[j];
						p = j;
					}
				}
				wfout << adj[i] << ' ' << index_to_y[p] << endl;
			}
			wfout << endl;
		}
		wfin.close();
		wfout.close();
	}

	void test_2(int which) {
		string input = dataset[which] + Dir + "test.in";
		string output = dataset[which] + Dir + "test.p5.out";
		wifstream wfin(input.c_str());
		wofstream wfout(output.c_str());
		if (!wfin) {
			cerr << "E3: File not found!" << endl;
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
			vec_array res;
			viterbi(adj, res);
			int n = (int)adj.size();
			double vote[Num_y];
			for (int i = 0; i < n; ++i) {
				memset(vote, 0, sizeof vote);
				for (int j = 0; j < Num_k; ++j) {
					int now = res.a[j][n - i - 1];
					vote[now] += weight[j];
				}
				double best = -1;
				int p = -1;
				for (int j = 0; j < Num_y; ++j) {
					if (vote[j] > best) {
						best = vote[j];
						p = j;
					}
				}
				wfout << adj[i] << ' ' << index_to_y[p] << endl;
			}
			wfout << endl;
		}
		wfin.close();
		wfout.close();
	}

public:
	Part_5() {
		pre();
	}

	void work() {
		for (int i = 0; i < Num_ds; ++i) {
			train(i);
			clog << dataset[i] << " train finished." << endl;
			test(i);
			clog << dataset[i] << " develop finished." << endl;
			test_2(i);
			clog << dataset[i] << " test finished." << endl;
		}
	}
};

int main()
{
	Part_5 p5;
	p5.work();
	return 0;
}
