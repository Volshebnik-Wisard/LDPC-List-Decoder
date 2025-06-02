#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

using namespace std;


// ������������ ����� ����������� ���� � ����� ������� ���� LDPC
class CheckNode {
private:
	map<int, double> receivedMessage; // �����, �������� ��������� �� ������������ ����� ����������

public:
	double calcMessage(int to) // ��������� ��������� ��������� � �������������� �������� tanh/atanh
	{
		double product = 1.0;
		for (const auto& [index, message] : receivedMessage) {
			if (index == to) continue;
			if (isinf(message) {
				if (message < 0) product *= -1;
				continue;
			}
			product *= tanh(message / 2);
		}
		return 2 * atanh(product);
	}

	// ��������� ��������� (��������� �������� ��������� (����))
	void receiveMessage(int from, double message) {
		receivedMessage[from] = message;
	}

	// �������� ��� ���������� ���������
	void clear() {
		receivedMessage.clear();
	}
};

// ������������ ����� ���������� ����� ���������� � ����������� ������
struct Edge {
	int variableNodeIndex; // ������ ���� ����������, ������������ ���� ������.
	int checkNodeIndex; // ������ ���� �������� (�����������), ������������ ���� ������.
};

// ������������ ����� ���� � ����� ������� LDPC
class VariableNode {
private:
	double channelLLR = 0.0; // ��������� ��������� ������������� ��������� �� ������
	map<int, double> receivedMessage; // �����, �������� ��������� �� ������������ ����������� ����� (���������� �����)
	bool isFrozen = false; // ����, �����������, ��������� �� ��� (���������)

public:
	//-������������ ���� : ���������� `+Inf` (������������� ��� `0`).
	//- ��������������: ���������� `ChannelLLR`.

	// ������������� ������������ ������ (��� � �������� �����)
	void setIsFrozen(bool frozen) {
		isFrozen = frozen;
	}

	// ���������� ��������� LLR (Inf ��� ������������ �����)
	double calcInitialMessage() {
		if (isFrozen) return INFINITY;
		return channelLLR;
	}

	// ��������� ��������� ��������� ����� ������������ �������� LLR
	/*
	1. ��������� `ChannelLLR` � ��� �������� LLR (�������� ������� ����).
    2. ���� �����-���� �������� ��������� ����� `�Inf`, �������������� ��� �������� (������ ����������).
       ������������ ����: ������ ���������� `+Inf` (�������������� ����������).
     */
	double calcMessage(int to) {
		if (isFrozen) return INFINITY;

		double sum = channelLLR;
		for (const auto& [index, message] : receivedMessage) {
			if (index == to) continue;
			if (isinf(message)) return message;
			sum += message;
		}
		return sum;
	}

	// ��������� �������� ���������
	void receiveMessage(int from, double message) {
		receivedMessage[from] = message;
	}

	// ��������� ������������� LLR ��� ������ ����� ����� ������������ ���� �������� ��������� � `ChannelLLR`.
	double marginalize() {
		return calcMessage(-1); // -1 ��������� �� ���������� �������� ����
	}

	// ��������� �������� ���� �� ������ LLR (����������� �������� LLR � ������� �������� (0/1))
	// ������� ������� (`0` ��� `1`)
	int estimateSendBit() {
		double llr = marginalize();
		if (llr > 0) return 0; //������������ ���� `0`
		if (llr < 0) return 1; //������������ ���� `1`

		// ��������� �������� ��� ������� LLR
		static random_device rd;
		static mt19937 gen(rd());
		uniform_int_distribution<> dis(0, 1);
		return dis(gen);
	}

	// �������� ��������� ����
	void clear() {
		channelLLR = 0.0;
		receivedMessage.clear();
	}

	// ���������� ����� LLR
	void setChannelLLR(double llr) {
		channelLLR = llr;
	}
};

// ��� LDPC ������������ ����� ������ ��������� ���� LDPC
class LDPCCode {
private:
	int codeLength; // ����� ����� �������� ����� (���������� ���������� �����)
	vector<Edge> edges; // ����� ����� ������ (������ ���� �����, ������������ ����� ����� ����������� � ������������ ������.)
	vector<int> informationBitIndexes; // ������� �������������� ����� (���������� �����, ����������� �������������� ���� (��������������))
	vector<int> frozenBitIndexes; // ������� ������������ �����
	vector<VariableNode> variableNodes; // ������ ���������� �����
	vector<CheckNode> checkNodes; // ������ ����������� �����
	const int decodeIteration = 40; // ���-�� ��������

	// ���������, ��������� �� ��� �������� �� �������� (������������� �� ������� ������� ����� ���� ������������ ��������)
	/*
	1. ��������� ���� ��� ���� ����� ���������� � ������� `Marginalize()`.
    2. ��������� ����, ������������ � ������� ������������ ���� (�� ������ 2).
    3. ���������� `true`, ������ ���� ��� ����� ����� `0` (��������).
	*/
	bool isSatisfyAllChecks() {
		vector<int> estimates(codeLength);
		for (int i = 0; i < codeLength; i++) {
			double llr = variableNodes[i].marginalize();
			if (llr == 0) return false;
			estimates[i] = (llr < 0) ? 1 : 0;
		}

		vector<int> checks(checkNodes.size(), 0);
		for (const auto& edge : edges) {
			checks[edge.checkNodeIndex] += estimates[edge.variableNodeIndex];
		}

		for (int c : checks) {
			if (c % 2 != 0) return false;
		}
		return true;
	}

	// ��������� �������� �������� ���������
	void executeMessagePassing(const vector<double>& channelOutputs) {
		// ��������������� ���������� ���� � ������� `ChannelLLR`
		for (int i = 0; i < codeLength; i++) {
			variableNodes[i].setChannelLLR(channelOutputs[i]);
		}

		// ������ �������� ��������: ���� ���������� ���������� ��������� ��������� ����� ��������.
		for (int i = 0; i < edges.size(); i++) {
			const auto& edge = edges[i];
			double message = variableNodes[edge.variableNodeIndex].calcInitialMessage();
			checkNodes[edge.checkNodeIndex].receiveMessage(i, message);
		}

		// �������� ���� �������� ���������
		for (int iter = 0; iter < decodeIteration; iter++) {
			// ���� �������� ��������� � ���������� ��������� ������� ����� ����������.
			for (int i = 0; i < edges.size(); i++) {
				const auto& edge = edges[i];
				double message = checkNodes[edge.checkNodeIndex].calcMessage(i);
				variableNodes[edge.variableNodeIndex].receiveMessage(i, message);
			}

			// ���� ���������� ��������� � ���������� ����� ���������.
			for (int i = 0; i < edges.size(); i++) {
				const auto& edge = edges[i];
				double message = variableNodes[edge.variableNodeIndex].calcMessage(i);
				checkNodes[edge.checkNodeIndex].receiveMessage(i, message);
			}

			// ��������� ����������, ���� ��� �������� ����� ���������
			if (isSatisfyAllChecks()) break;
		}
	}

public:
	// ������������� �������� ������ ������������ ������ (��������� �������������� ���� (��������� `EstimateSendBit()`) ������ ��� �������������� �����.)
	vector<int> decode(const vector<double>& channelOutputs) {
		executeMessagePassing(channelOutputs);
		vector<int> decoded;
		for (int index : informationBitIndexes) {
			decoded.push_back(variableNodes[index].estimateSendBit());
		}
		return decoded;
	}

	// ������������� ������ ��� ������������� ����� (���������� ��������� ������� ����-���������� ��� ������������� �����.)
	// ��������� ���������� ���� �� �������� LLR � �������������� �� ��� �������� �����������.
	vector<vector<int>> listDecode(const vector<double>& channelOutputs, int listSize) {
		// ���������� �������� �������� ���� (��������� � `0` LLR) ����� �������������� �����.
		executeMessagePassing(channelOutputs);
		// ���������� ���������� `listSize`, ������������ ������������� ��� ����.
		int ambiguousBitCount = floor(log2(listSize));
		vector<pair<int, double>> llrs;
		for (int index : informationBitIndexes) {
			double llr = variableNodes[index].marginalize();
			llrs.emplace_back(index, llr);
		}
		// ���������� �� �������� LLR (������� �������� ��������)
		sort(llrs.begin(), llrs.end(), [](const auto& a, const auto& b) {
			return abs(a.second) < abs(b.second);
			});

		// ��������� ����������� ��������������� �������
		vector<int> uniqueDecoded;
		for (int index : informationBitIndexes) {
			uniqueDecoded.push_back(variableNodes[index].estimateSendBit());
		}

		vector<vector<int>> listDecoded = { uniqueDecoded };

		// ���������� ������ ������� ����, ������� �������� � ������������ ��������.
		for (int i = 0; i < ambiguousBitCount; i++) {
			auto [index, _] = llrs[i];
			vector<vector<int>> temp;
			for (const auto& v : listDecoded) {
				vector<int> inverted = v;
				inverted[index] = 1 - inverted[index];
				temp.push_back(inverted);
			}
			listDecoded.insert(listDecoded.end(), temp.begin(), temp.end());
		}

		return listDecoded;
	}

	// ���������� �������� �������� �����
	double getRate() const {
		return static_cast<double>(informationBitIndexes.size()) / getRealCodeLength();
	}

	// ���������� ����������� �������� ��� ������������� ������
	double getListRate(int listSize) const {
		int ambiguousBitCount = floor(log2(listSize));
		return static_cast<double>(informationBitIndexes.size() - ambiguousBitCount) / getRealCodeLength();
	}

	// ���������� ����������� ����� �������� �����
	int getRealCodeLength() const {
		return codeLength - frozenBitIndexes.size();
	}

	// �������� ���������� �������� LDPC-����
		static LDPCCode constructCode(
		int originalCodeLength,
		int informationBitSize,
		int variableNodeDegree,
		int checkNodeDegree) {

		LDPCCode code;
		code.codeLength = originalCodeLength;

		// ������������� �������������� � ������������ ������� �������
		for (int i = 0; i < informationBitSize; i++) {
			code.informationBitIndexes.push_back(i);
		}

		int originalInfoBitSize = originalCodeLength -
			(originalCodeLength * variableNodeDegree) / checkNodeDegree;

		for (int i = informationBitSize; i < originalInfoBitSize; i++) {
			code.frozenBitIndexes.push_back(i);
		}

		// �������� �����
		code.edges = createRandomEdges(originalCodeLength, variableNodeDegree, checkNodeDegree);

		// ���������������� ����
		code.variableNodes.resize(originalCodeLength);
		code.checkNodes.resize((originalCodeLength * variableNodeDegree) / checkNodeDegree);

		// ����� ������������ ���
		for (int index : code.frozenBitIndexes) {
			code.variableNodes[index].setIsFrozen(true);
		}

		return code;
	}

private:
	// ������ ��������� ����� ��� ����� �������
	// ���� ������� �������� ���������, �� ���������� (������������� �������).
	static vector<Edge> createRandomEdges(int length, int variableNodeDegree, int checkNodeDegree) {
		if (length * variableNodeDegree % checkNodeDegree != 0) {
			throw invalid_argument("Invalid length and degree combination");
		}

		random_device rd;
		mt19937 gen(rd());

		vector<Edge> edges(length * variableNodeDegree);
		vector<int> temp(length * variableNodeDegree);

		// Initialize with variable node indices
		for (int i = 0; i < temp.size(); i++) {
			temp[i] = i / variableNodeDegree;
		}

		// Shuffle the connections
		shuffle(temp.begin(), temp.end(), gen);

		// Create edges
		for (int k = 0; k < temp.size(); k++) {
			edges[k] = Edge{ temp[k], k / checkNodeDegree };
		}

		return edges;
	}
};