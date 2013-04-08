#include "gspConsoleInterface.h"
#include "gspDataGraphExpand.h"
#include "OneDimKMean.h"
#include "gspFileObj.h"
#include "gspFileLbe.h"
#include "gspChWordGram5.h"
#include <list>
#include <vector>
#include <string>

#define ClassNum 6
#define Threshold 0.01f
#define MaxIterNum 1000
#define SIL 65536

using namespace std;

char *g_strHelp = 
"Usage: %s -train=%%s(filelist，训练文件列表) -learn=%%s(filelist，待标注文件列表)"
"-learnLbe=%%s(Lbefilelist)\n";

inline int GetCNSliceNum(const CDataGraph &p_CN);
bool ZigCNLbeAlign(CDataGraph &p_CN, CSet &p_LbeContent);//用于对cn中的弧（DATA_NODE）进行标注，将标注信息记录在每个slice的开始节点（虚节点）的hook中
int main(int p_nArgc, char **p_ppcArgv)
{
	if (p_nArgc < 3)
	{
		printf(g_strHelp, p_ppcArgv[0]);
		return 1;
	}

	GPATH strTrainFilelist = "";
	GPATH strLearnLbeFilelist = "";
	GPATH strLearnFilelist = "";

	inputCommand("-train=%s", strTrainFilelist);
	inputCommand("-learnLbe=%s", strLearnLbeFilelist);
	inputCommand("-learn=%s", strLearnFilelist);

	outputResult("-train=%s\n", strTrainFilelist);
	outputResult("-learnLbe=%s\n", strLearnLbeFilelist);
	outputResult("-learn=%s\n", strLearnFilelist);

	vector<string> vecTrainFile;
	vector<string> vecLearnLbeFile;
	vector<string> vecLearnFile;
	GPATH strTemp = "";

	FILE *fpTrainList = fopen(strTrainFilelist, "r");
	if (fpTrainList == NULL)
	{
		printf("Open file %s error!\n", strTrainFilelist);
		return 1;
	}
	while(true)
	{
		fscanf(fpTrainList, "%s", strTemp);
		if(feof(fpTrainList)) break;
		vecTrainFile.push_back(string(strTemp));
	}
	fclose(fpTrainList);

	FILE *fpLearnList = fopen(strLearnFilelist, "r");
	if (fpLearnList == NULL)
	{
		printf("Open file %s error!\n", strLearnFilelist);
		return 1;
	}
	while(true)
	{
		fscanf(fpLearnList, "%s", strTemp);
		if(feof(fpLearnList)) break;
		vecLearnFile.push_back(string(strTemp));
	}
	fclose(fpLearnList);

	FILE *fpLearnLbeFilelist = fopen(strLearnLbeFilelist, "r");
	if (fpLearnLbeFilelist == NULL)
	{
		printf("Open file %s error!\n", strLearnLbeFilelist);
		return 1;
	}
	while(true)
	{
		fscanf(fpLearnLbeFilelist, "%s", strTemp);
		if(feof(fpLearnLbeFilelist)) break;
		vecLearnLbeFile.push_back(string(strTemp));
	}
	fclose(fpLearnLbeFilelist);

	int nTrainFileNum = vecTrainFile.size();
	int nLearnLbeFileNum = vecLearnLbeFile.size();
	int nLearnFileNum = vecLearnFile.size();
	if(nLearnLbeFileNum != nLearnFileNum)
	{
		printf("filelist %s and %s do not match\n", strLearnFilelist, strLearnLbeFilelist);
		return 1;
	}

	//Training
	printf("Starting to train...\n");
	int nTrainElemNum = 0;
	list<ElemType> listTrainElemBuff;

	CObjFile FileObj;
	CDataGraph CNGraph;
	FileObj.m_pObj = &CNGraph;
	FILE *fpTrainFile = NULL;
	for (int n = 0; n < nTrainFileNum; ++ n)
	{
		if ((fpTrainFile = fopen(vecTrainFile[n].c_str(), "rt")) == NULL)
		{
			printf("Open file %s error!\n", vecTrainFile[n].c_str());
			return 1;
		}

		FileObj.ReadHeadT(fpTrainFile);
		int nObjNum = FileObj.m_nObjNum;

		for (int nObjNo = 0; nObjNo < nObjNum; ++ nObjNo)
		{
			CNGraph.ResetDataGraph();
			CNGraph.m_nStateSegFlag = 1;
			FileObj.ReadOneT(fpTrainFile);
			DATA_NODE *pCurrNode = CNGraph.m_pdnodeArray;
			for (int nNodeNo = 0; nNodeNo < CNGraph.m_nNodesNum; ++ nNodeNo)
			{
				if(pCurrNode->nNodeContentId > 0)
					listTrainElemBuff.push_back(pCurrNode->fConfidence);
				++ pCurrNode;
			}
		}
		fclose(fpTrainFile);
	}

	nTrainElemNum = listTrainElemBuff.size();
	ElemType *pTrainElemBuff = new ElemType[nTrainElemNum];
	list<ElemType>::iterator listiter = listTrainElemBuff.begin();
	for (int nElemNo = 0; nElemNo < nTrainElemNum; ++ nElemNo)
	{
		pTrainElemBuff[nElemNo] = *listiter;
		++ listiter;
	}
	listTrainElemBuff.clear();
	ElemType *pClassCentre = new ElemType[ClassNum];
	OneDimKMeanTrain(pTrainElemBuff, nTrainElemNum, ClassNum, MaxIterNum, Threshold, pClassCentre);
	delete []pTrainElemBuff;
	printf("Training ends!\nStarting to learn...\n");
	FILE *fpLearnFile = NULL;
	FILE *fpLearnLbe = NULL;
	CDataGraph LearnCNGraph;
	FileObj.m_pObj = &LearnCNGraph;
	CLbeFile fileLbe;

	FILE *fpConf = NULL;
	FILE *fpLmbb = NULL;
	FILE *fpConfLmbb = NULL;

	for (int n = 0; n < nLearnFileNum; ++ n)
	{
		if ((fpLearnFile = fopen(vecLearnFile[n].c_str(), "rt")) == NULL)
		{
			printf("Open file %s error!\n", vecLearnFile[n].c_str());
			return 1;
		}

		if ((fpLearnLbe = fopen(vecLearnLbeFile[n].c_str(), "rt")) == NULL)
		{
			printf("Open file %s error!\n", vecLearnLbeFile[n].c_str());
			return 1;
		}

		FileObj.ReadHeadT(fpLearnFile);
		int nObjNum = FileObj.m_nObjNum;
		fileLbe.ReadHeadT(fpLearnLbe);
		int nSentNum = 0;
		fileLbe.GetSegNum(nSentNum, fpLearnLbe);
		if(nSentNum != nObjNum)
		{
			printf("CN graph file %s and lbe file %s do not match!\n", 
				vecLearnFile[n].c_str(), vecLearnLbeFile[n].c_str());
			return 1;
		}
		CSet LbeZiIntSet;
		int nSylNum = 0;
		if((fpConf = fopen((vecLearnFile[n] + ".conf").c_str(), "wt")) == NULL)
		{
			printf("Open file %s error!\n", (vecLearnFile[n] + ".conf").c_str());
			return 1;
		}

		if((fpLmbb = fopen((vecLearnFile[n] + ".lmbb").c_str(), "wt")) == NULL)
		{
			printf("Open file %s error!\n", (vecLearnFile[n] + ".lmbb").c_str());
			return 1;
		}

		if((fpConfLmbb = fopen((vecLearnFile[n] + ".conf.lmbb").c_str(), "wt")) == NULL)
		{
			printf("Open file %s error!\n", (vecLearnFile[n] + ".conf.lmbb").c_str());
			return 1;
		}

		for (int nObjNo = 0; nObjNo < nObjNum; ++ nObjNo)
		{
			fileLbe.ReadLineT(fpLearnLbe);
			fileLbe.GetSylNum(nSylNum);
			fileLbe.GetZiInt(nSylNum, LbeZiIntSet);
			LearnCNGraph.ResetDataGraph();
			LearnCNGraph.m_nStateSegFlag = 1;
			FileObj.ReadOneT(fpLearnFile);
			//为cn添加标注，标注信息由各个slice的开始节点的hook记录（nChild或者-1（other））
			ZigCNLbeAlign(LearnCNGraph, LbeZiIntSet);
			//离散后验概率
			ElemType *pfPosterior = new ElemType[LearnCNGraph.m_nNodesNum];
			int *pnDiscreteConf = new int[LearnCNGraph.m_nNodesNum];
			for (int nNode = 0; nNode < LearnCNGraph.m_nNodesNum; ++nNode)
			{
				pfPosterior[nNode] = LearnCNGraph.m_pdnodeArray[nNode].fConfidence;
			}

			OneDimKMeanClassfy(pfPosterior, LearnCNGraph.m_nNodesNum, ClassNum, pClassCentre, pnDiscreteConf);
			for (int nNode = 0; nNode < LearnCNGraph.m_nExitNode; ++nNode)
			{
				for (int nChild = 0; nChild < LearnCNGraph.m_pdnodeArray[nNode].nChildrenNum; ++nChild)
				{
					if(LearnCNGraph.m_pdnodeArray[LearnCNGraph.m_pdnodeArray[nNode].pnChildren[nChild]].nNodeContentId == SIL)
						continue;//跳过删除符
					fprintf(fpConf, "%d\t", pnDiscreteConf[LearnCNGraph.m_pdnodeArray[nNode].pnChildren[nChild]]);
					fprintf(fpLmbb, "%d\t", LearnCNGraph.m_pdnodeArray[LearnCNGraph.m_pdnodeArray[nNode].pnChildren[nChild]].pnStateSeg[0]);
					fprintf(fpConfLmbb, "%d%d\t", pnDiscreteConf[LearnCNGraph.m_pdnodeArray[nNode].pnChildren[nChild]]
					, LearnCNGraph.m_pdnodeArray[LearnCNGraph.m_pdnodeArray[nNode].pnChildren[nChild]].pnStateSeg[0]);
				}
				int nLbe = (int)(LearnCNGraph.m_pdnodeArray[nNode].hook);
				if(nLbe == -1)
				{
					fprintf(fpConf, "other\n");
					fprintf(fpLmbb, "other\n");
					fprintf(fpConfLmbb, "other\n");
				}
				else
				{
					fprintf(fpConf, "%d\n", pnDiscreteConf[LearnCNGraph.m_pdnodeArray[nNode].pnChildren[nLbe]]);
					fprintf(fpLmbb, "%d\n", LearnCNGraph.m_pdnodeArray[LearnCNGraph.m_pdnodeArray[nNode].pnChildren[nLbe]].pnStateSeg[0]);
					fprintf(fpConfLmbb, "%d%d\n", pnDiscreteConf[LearnCNGraph.m_pdnodeArray[nNode].pnChildren[nLbe]],
						LearnCNGraph.m_pdnodeArray[LearnCNGraph.m_pdnodeArray[nNode].pnChildren[nLbe]].pnStateSeg[0]);
				}
			}
			fprintf(fpConf, "\n");
			fprintf(fpLmbb, "\n");
			fprintf(fpConfLmbb, "\n");

			delete []pnDiscreteConf;
			delete []pfPosterior;
		}

		fclose(fpConf);
		fclose(fpLmbb);
		fclose(fpConfLmbb);
		fclose(fpLearnFile);
		fclose(fpLearnLbe);
	}
	delete []pClassCentre;
	return 0;
}

int GetCNSliceNum(const CDataGraph &p_CN)
{
	return p_CN.m_nExitNode;
}

bool ZigCNLbeAlign(CDataGraph &p_CN, CSet &p_LbeContent)
{
	int nCNSliceNum = GetCNSliceNum(p_CN);
	int nLbeSize = p_LbeContent.m_nNum;
	int nRowNum = nCNSliceNum + 1;
	int nColumnNum = nLbeSize + 1;
	if(nCNSliceNum <= 0 || nLbeSize <=0) return false;
	unsigned *pnDistArray = new unsigned[nRowNum * nColumnNum];
	pnDistArray[0] = 0;
	char *pchPathRecord = new char[nRowNum * nColumnNum];//0,1,2：用于记录最优路径

	int nColumn = 0;
	int nRow = 0;
	for (nColumn = 0; nColumn < nColumnNum; ++nColumn)
	{
		pnDistArray[nColumn] = nColumn;
		pchPathRecord[nColumn] = 0;
	}

	int nRowPos = 0;
	for (nRow = 1; nRow < nRowNum; ++nRow)
	{
		nRowPos += nColumnNum;
		pnDistArray[nRowPos] = nRow;
		pchPathRecord[nRowPos] = 2;
	}

	nRowPos = 0;
	for (nRow = 1; nRow < nRowNum; ++nRow)
	{
		nRowPos += nColumnNum;
		for (nColumn = 1; nColumn < nColumnNum; ++nColumn)
		{
			unsigned nDist0 = pnDistArray[nRowPos + nColumn - 1] + 1;
			unsigned nDist2 = pnDistArray[nRowPos - nColumnNum + nColumn] + 1;
			unsigned nDist1 = 0;
			int nLbeZig = *(int *)p_LbeContent.Get(nColumn - 1);
			unsigned nSubDist = 1;
			for (int nChild = 0; nChild < p_CN.m_pdnodeArray[nRow - 1].nChildrenNum; ++nChild)
			{
				if(p_CN.m_pdnodeArray[p_CN.m_pdnodeArray[nRow - 1].pnChildren[nChild]].nNodeContentId == nLbeZig)
				{
					nSubDist = 0;
					break;
				}
			}

			nDist1 = pnDistArray[nRowPos - nColumnNum + nColumn - 1] + nSubDist;

			if (nDist0 < nDist1)
			{
				if (nDist0 < nDist2)
				{
					pnDistArray[nRowPos + nColumn] = nDist0;
					pchPathRecord[nRowPos + nColumn] = 0;
				}

				else
				{
					pnDistArray[nRowPos + nColumn] = nDist2;
					pchPathRecord[nRowPos + nColumn] = 2;
				}
			}

			else
			{
				if (nDist1 < nDist2)
				{
					pnDistArray[nRowPos + nColumn] = nDist1;
					pchPathRecord[nRowPos + nColumn] = 1;
				}

				else
				{
					pnDistArray[nRowPos + nColumn] = nDist2;
					pchPathRecord[nRowPos + nColumn] = 2;
				}
			}
		}
	}

	int nPos = nRowNum * nColumnNum - 1;
	nRow = nRowNum - 1;
	nColumn = nColumnNum - 1;
	int nLbeZig = 0;
	while(nRow > 0)
	{
		switch (pchPathRecord[nPos])
		{
		case 0:
			nPos = nPos - 1;
			--nColumn;
			break;
		case 1:
			p_CN.m_pdnodeArray[nRow - 1].hook = (void *)-1;
			nLbeZig = *(int *)p_LbeContent.Get(nColumn - 1);
			for (int nChild = 0; nChild < p_CN.m_pdnodeArray[nRow - 1].nChildrenNum; ++nChild)
			{
				if(p_CN.m_pdnodeArray[p_CN.m_pdnodeArray[nRow - 1].pnChildren[nChild]].nNodeContentId == nLbeZig)
				{
					p_CN.m_pdnodeArray[nRow - 1].hook = (void *)nChild;
					break;
				}
			}
			--nColumn;
			--nRow;
			nPos = nPos - nColumnNum - 1;
			break;
		case 2:
			p_CN.m_pdnodeArray[nRow - 1].hook = (void *)-1;
			--nRow;
			nPos = nPos - nColumnNum;
			break;
		default:
			fprintf(stderr, "Path record error!\n");
			delete []pnDistArray;
			delete []pchPathRecord;
			return false;
		}
	}

	delete []pchPathRecord;
	delete []pnDistArray;
	return true;
}