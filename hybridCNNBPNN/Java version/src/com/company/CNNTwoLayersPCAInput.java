package com.company;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import Jama.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.lang.model.util.ElementScanner14;


public class CNNTwoLayersPCAInput {
	
	public int rowNumber = 159;
	public int colNumber = 159;
	public int pcaDim = 159;
	public double[][] avgMatrix = new double[rowNumber][colNumber];
	public double[][] coVarianceMatrix = new double[rowNumber][colNumber];
	public double[][] eigenVectors = new double[colNumber][pcaDim];
	
	public int matrixWidth = pcaDim;
	public int matrixHeight = rowNumber;
	public int convLayer1Size = 5;
	public int convLayer1MapWidth = matrixWidth - convLayer1Size + 1;
	public int convLayer1MapHeight = matrixHeight - convLayer1Size + 1;
	public int poolLayer1Size = 2;
	public int poolLayer1MapWidth = convLayer1MapWidth / poolLayer1Size;
	public int poolLayer1MapHeight = convLayer1MapHeight / poolLayer1Size;
	public int convLayer1Num = 5;
	public int poolLayer1Num = convLayer1Num;
	public int convLayer2Size = 5;
	public int convLayer2MapWidth = poolLayer1MapWidth - convLayer2Size + 1;
	public int convLayer2MapHeight = poolLayer1MapHeight - convLayer2Size + 1;
	public int poolLayer2Size = 2;
	public int poolLayer2MapWidth = convLayer2MapWidth / poolLayer2Size;
	public int poolLayer2MapHeight = convLayer2MapHeight / poolLayer2Size;
	public int convLayer2Num = poolLayer1Num;
	public int poolLayer2Num = convLayer2Num;
	public int fullNum = poolLayer2Num * poolLayer2MapWidth * poolLayer2MapHeight;
	public int unitNum = fullNum;
	public int outputNum = 2;
	public int valueNum = outputNum;
	
	public double[][] matrix = new double[matrixHeight][matrixWidth];
	public double[][][] convLayer1Core = new double[convLayer1Num][convLayer1Size][convLayer1Size];
	public double[][][] convLayer1Map = new double[convLayer1Num][convLayer1MapHeight][convLayer1MapWidth];
	public double[][][] poolLayer1Core = new double[poolLayer1Num][poolLayer1Size][poolLayer1Size];
	public double[][][] poolLayer1Map = new double[poolLayer1Num][poolLayer1MapHeight][poolLayer1MapWidth];
	public double[][][] convLayer2Core = new double[convLayer2Num][convLayer2Size][convLayer2Size];
	public double[][][][] convLayer2Map = new double[poolLayer1Num][convLayer2Num][convLayer2MapHeight][convLayer2MapWidth];
	public double[][][] poolLayer2Core = new double[poolLayer2Num][poolLayer2Size][poolLayer2Size];
	public double[][][] poolLayer2Map = new double[poolLayer2Num][poolLayer2MapHeight][poolLayer2MapWidth];
	public double[] fullConnect = new double[fullNum];
	public double[] unit = new double[unitNum];
	public double[] output = new double[outputNum];
	public double[] value = new double[valueNum];
	
	public double[] fullConnectToUnitWeight = new double[fullNum];
	public double[][] unitToOutputWeight = new double[unitNum][outputNum];
	
	public String malwareFolderPath = "E:\\Dataset(A feature-hybrid malware variants detection )\\malicious-png\\";
	public String benignFolderPath = "E:\\Dataset(A feature-hybrid malware variants detection )\\benign-png\\";
	
	public double[][] readMatrix(File file) {
		try {
			double[][] matrix = new double[rowNumber][colNumber];
			//read file
			if (file.exists())
				System.out.println("File exists!");
			else
				System.out.println("File does not exist!");
			BufferedImage image = ImageIO.read(file);
			int width = image.getWidth();
			int height = image.getHeight();
			for(int i=0;i<width;i++) {
				for(int j=0;j<height;j++) {
					int pixel = image.getRGB(i, j);
					int[] rgb = new int[3];
					rgb[0] = (pixel & 0xff0000) >> 16;
					matrix[j][i] = rgb[0];
				}
			}
			return matrix;
		}catch(Exception e) {
			e.printStackTrace();
			return null;
		}
	}
	
	public double[][] transpositionMatrix(double data[][],int row,int col) {
		double[][] rsData = new double[col][row];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				rsData[j][i] = data[i][j];
			}
		}
		return rsData;
	}

	public double[][] multiplyMatrix(double data1[][],int row1,int col1,double data2[][],int row2,int col2) {
		if(row2 != col1) {
			return null;
		}
		double[][] rsData = new double[row1][col2];
		for (int i = 0; i < row1; i++) {
			for (int j = 0; j < col2; j++) {
				for(int k = 0; k < col1; k++) {
					rsData[i][j] += data1[i][k] * data2[k][j];
				}
			}
		}
		return rsData;
	}
	
	public void averageMatrix(int malwareTrainNumber,int benignTrainNumber) {
		try {
			File folder = new File(malwareFolderPath);
			File[] fileList = folder.listFiles();
			for(int n=0;n<fileList.length && n<malwareTrainNumber;n++) {
				File file = fileList[n];
				double[][] trainInput = readMatrix(file);
				for(int i=0;i<rowNumber;i++) {
					for(int j=0;j<colNumber;j++) {
						avgMatrix[i][j] += trainInput[i][j];
					}
				}
			}
			folder = new File(benignFolderPath);
			fileList = folder.listFiles();
			for(int n=0;n<fileList.length && n<benignTrainNumber;n++) {
				File file = fileList[n];
				double[][] trainInput = readMatrix(file);
				for(int i=0;i<rowNumber;i++) {
					for(int j=0;j<colNumber;j++) {
						avgMatrix[i][j] += trainInput[i][j];
					}
				}
			}
			for(int i=0;i<rowNumber;i++) {
				for(int j=0;j<colNumber;j++) {
					avgMatrix[i][j] = avgMatrix[i][j] / (malwareTrainNumber + benignTrainNumber);
				}
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void coVarianceMatrix(int malwareTrainNumber,int benignTrainNumber) {
		try {
			File folder = new File(malwareFolderPath);
			File[] fileList = folder.listFiles();
			for(int n=0;n<fileList.length && n<malwareTrainNumber;n++) {
				File file = fileList[n];
				double[][] trainInput = readMatrix(file);
				double[][] variance = new double[rowNumber][colNumber];
				for(int i=0;i<rowNumber;i++) {
					for(int j=0;j<colNumber;j++) {
						variance[i][j] = trainInput[i][j] - avgMatrix[i][j];
					}
				}
				double[][] tempCoVariance = multiplyMatrix(transpositionMatrix(variance,rowNumber,colNumber),colNumber,rowNumber,variance,rowNumber,colNumber);
				for(int i=0;i<colNumber;i++) {
					for(int j=0;j<colNumber;j++) {
						coVarianceMatrix[i][j] += tempCoVariance[i][j];
					}
				}
			}
			folder = new File(benignFolderPath);
			fileList = folder.listFiles();
			for(int n=0;n<fileList.length && n<benignTrainNumber;n++) {
				File file = fileList[n];
				double[][] trainInput = readMatrix(file);
				double[][] variance = new double[rowNumber][colNumber];
				for(int i=0;i<rowNumber;i++) {
					for(int j=0;j<colNumber;j++) {
						variance[i][j] = trainInput[i][j] - avgMatrix[i][j];
					}
				}
				double[][] tempCoVariance = multiplyMatrix(transpositionMatrix(variance,rowNumber,colNumber),colNumber,rowNumber,variance,rowNumber,colNumber);
				for(int i=0;i<colNumber;i++) {
					for(int j=0;j<colNumber;j++) {
						coVarianceMatrix[i][j] += tempCoVariance[i][j];
					}
				}
			}
			for(int i=0;i<colNumber;i++) {
				for(int j=0;j<colNumber;j++) {
					coVarianceMatrix[i][j] = coVarianceMatrix[i][j] / (malwareTrainNumber + benignTrainNumber);
				}
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void getEigen() {
		try {
			Matrix e = new Matrix(coVarianceMatrix);
			for(int i=0;i<rowNumber;i++) {
				for(int j=0;j<colNumber;j++) {
					eigenVectors[i][j]=e.eig().getV().get(i,j);}}
			for(int i=0;i<colNumber;i++) {
				for(int j=colNumber-pcaDim;j<colNumber;j++) {
					eigenVectors[i][j-(colNumber-pcaDim)] = eigenVectors[i][j];
				}
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void randomConvCore() {
		Random rand = new Random();
		for(int n=0;n<convLayer1Num;n++) {
			for(int i=0;i<convLayer1Size;i++) {
				for(int j=0;j<convLayer1Size;j++) {
					double temp = rand.nextInt(1000);
					temp = temp / 1000;
					convLayer1Core[n][i][j]=temp;
				}
			}
		}
		for(int n=0;n<convLayer2Num;n++) {
			for(int i=0;i<convLayer2Size;i++) {
				for(int j=0;j<convLayer2Size;j++) {
					double temp = rand.nextInt(1000);
					temp = temp / 1000;
					convLayer2Core[n][i][j]=temp;
				}
			}
		}
	}
	
	public void initPoolCore() {
		for(int n=0;n<poolLayer1Num;n++) {
			for(int i=0;i<poolLayer1Size;i++) {
				for(int j=0;j<poolLayer1Size;j++) {
					poolLayer1Core[n][i][j]=1.0 / (poolLayer1Size * poolLayer1Size);
				}
			}
		}
		for(int n=0;n<poolLayer2Num;n++) {
			for(int i=0;i<poolLayer2Size;i++) {
				for(int j=0;j<poolLayer2Size;j++) {
					poolLayer2Core[n][i][j]=1.0 / (poolLayer2Size * poolLayer2Size);
				}
			}
		}
	}
	
	public void initFullConnectToUnitWeight() {
		for(int i=0;i<fullNum;i++) {
			fullConnectToUnitWeight[i]=1.0;
		}
	}
	
	public void randomUnitToOutputWeight() {
		Random rand = new Random();
		for(int i=0;i<unitNum;i++) {
			for(int j=0;j<outputNum;j++) {
				double temp = rand.nextInt(1000);
				temp = temp / 1000;
				unitToOutputWeight[i][j]=temp;
			}
		}
	}
	
	public double getAveragePixel(int malwareTrainingNum, int benignTrainingNum) throws IOException {
		double sum = 0.0;
		double avg = 0.0;
		double total = 0;
		File folder = new File(malwareFolderPath);
		File[] fileList = folder.listFiles();
		for(int n=0;n<fileList.length && n<malwareTrainingNum;n++) {
			String fileName = malwareFolderPath+fileList[n].getName();
			File file = new File(fileName);
			double[][] trainInput = readMatrix(file);
			double[][] variance = new double[rowNumber][colNumber];
			for(int i=0;i<rowNumber;i++) {
				for(int j=0;j<colNumber;j++) {
					variance[i][j] = trainInput[i][j] - avgMatrix[i][j];
				}
			}
			double[][] eigenMatrix = multiplyMatrix(variance,rowNumber,colNumber,eigenVectors,colNumber,pcaDim);
			sum = 0.0;
			for(int i=0;i<rowNumber;i++) {
				for(int j=0;j<pcaDim;j++) {
					sum += eigenMatrix[i][j];
				}
			}
			avg += sum / (rowNumber * pcaDim);
			total++;
		}
		folder = new File(benignFolderPath);
		fileList = folder.listFiles();
		for(int n=0;n<fileList.length && n<benignTrainingNum;n++) {
			String fileName = benignFolderPath+fileList[n].getName();
			File file = new File(fileName);
			double[][] trainInput = readMatrix(file);
			double[][] variance = new double[rowNumber][colNumber];
			for(int i=0;i<rowNumber;i++) {
				for(int j=0;j<colNumber;j++) {
					variance[i][j] = trainInput[i][j] - avgMatrix[i][j];
				}
			}
			double[][] eigenMatrix = multiplyMatrix(variance,rowNumber,colNumber,eigenVectors,colNumber,pcaDim);
			sum = 0.0;
			for(int i=0;i<rowNumber;i++) {
				for(int j=0;j<pcaDim;j++) {
					sum += eigenMatrix[i][j];
				}
			}
			avg += sum / (rowNumber * pcaDim);
			total++;
		}
		avg = avg / total;
		return avg;
	}
	
	public void initMatrixAndValue(String filePathName,double averagePixel,int index) throws IOException {
		for(int i=0;i<valueNum;i++) {
			if(i==index) {
				value[i] = 1;
			}else {
				value[i] = 0;
			}
		}
		File file = new File(filePathName);
		double[][] trainInput = readMatrix(file);
		double[][] variance = new double[rowNumber][colNumber];
		for(int i=0;i<rowNumber;i++) {
			for(int j=0;j<colNumber;j++) {
				variance[i][j] = trainInput[i][j] - avgMatrix[i][j];
			}
		}
		double[][] eigenMatrix = multiplyMatrix(variance,rowNumber,colNumber,eigenVectors,colNumber,pcaDim);
		for(int i=0;i<rowNumber;i++) {
			for(int j=0;j<pcaDim;j++) {
				eigenMatrix[i][j] = eigenMatrix[i][j] - averagePixel;
				eigenMatrix[i][j] = eigenMatrix[i][j] / (rowNumber * pcaDim);
				matrix[i][j] = eigenMatrix[i][j];
			}
		}
	}
	
	public void convolutionMatrix() {
		for(int n=0;n<convLayer1Num;n++) {
			for(int h=0;h<convLayer1MapHeight;h++) {
				for(int w=0;w<convLayer1MapWidth;w++) {
					convLayer1Map[n][h][w] = 0;
					for(int i=0;i<convLayer1Size;i++) {
						for(int j=0;j<convLayer1Size;j++) {
							convLayer1Map[n][h][w] += matrix[h+i][w+j] * convLayer1Core[n][i][j];
						}
					}
					convLayer1Map[n][h][w] = convLayer1Map[n][h][w] / (convLayer1Size * convLayer1Size);
				}
			}
		}
	}
	
	public void poolingConvLayer1Map() {
		for(int n=0;n<poolLayer1Num;n++) {
			for(int h=0;h<poolLayer1MapHeight;h++) {
				for(int w=0;w<poolLayer1MapWidth;w++) {
					poolLayer1Map[n][h][w] = 0;
					for(int i=0;i<poolLayer1Size;i++) {
						for(int j=0;j<poolLayer1Size;j++) {
							poolLayer1Map[n][h][w] += convLayer1Map[n][2*h+i][2*w+j] * poolLayer1Core[n][i][j];
						}
					}
				}
			}
		}
	}
	
	public void convolutionLayer1() {
		for(int m=0;m<poolLayer1Num;m++) {
			for(int n=0;n<convLayer2Num;n++) {
				for(int h=0;h<convLayer2MapHeight;h++) {
					for(int w=0;w<convLayer2MapWidth;w++) {
						convLayer2Map[m][n][h][w] = 0;
						for(int i=0;i<convLayer2Size;i++) {
							for(int j=0;j<convLayer2Size;j++) {
								convLayer2Map[m][n][h][w] += poolLayer1Map[m][h+i][w+j] * convLayer2Core[n][i][j];
							}
						}
						convLayer2Map[m][n][h][w] = convLayer2Map[m][n][h][w] / (convLayer2Size * convLayer2Size);
					}
				}
			}
		}
	}
	
	public double[][][][] poolLayer1ToConvWeight = new double[poolLayer1Num][convLayer2Num][convLayer2MapHeight][convLayer2MapWidth];
	
	public void randomPoolLayer1ToConvWeight() {
		Random rand = new Random();
		for(int h=0;h<convLayer2MapHeight;h++) {
			for(int w=0;w<convLayer2MapWidth;w++) {
				for(int j=0;j<convLayer2Num;j++) {
					for(int i=0;i<poolLayer1Num;i++) {
						double temp = rand.nextInt(1000);
						temp = temp / 1000;
						poolLayer1ToConvWeight[i][j][h][w]=temp;
					}
				}
			}
		}
	}
	
	public double[][][] convLayer2WeightedMap = new double[convLayer2Num][convLayer2MapHeight][convLayer2MapWidth];
	
	public void layer1ToLayer2Weight() {
		for(int h=0;h<convLayer2MapHeight;h++) {
			for(int w=0;w<convLayer2MapWidth;w++) {
				for(int j=0;j<convLayer2Num;j++) {
					for(int i=0;i<poolLayer1Num;i++) {
						convLayer2WeightedMap[j][h][w] += convLayer2Map[i][j][h][w] * poolLayer1ToConvWeight[i][j][h][w];
					}
					convLayer2WeightedMap[j][h][w] = convLayer2WeightedMap[j][h][w] / poolLayer1Num;
				}
			}
		}
	}
	
	public void poolingConvLayer2Map() {
		for(int n=0;n<poolLayer2Num;n++) {
			for(int h=0;h<poolLayer2MapHeight;h++) {
				for(int w=0;w<poolLayer2MapWidth;w++) {
					poolLayer2Map[n][h][w] = 0;
					for(int i=0;i<poolLayer2Size;i++) {
						for(int j=0;j<poolLayer2Size;j++) {
							//poolLayer2Map[n][h][w] += convLayer2Map[n][2*h+i][2*w+j] * poolLayer2Core[n][i][j];
							poolLayer2Map[n][h][w] += convLayer2WeightedMap[n][2*h+i][2*w+j] * poolLayer2Core[n][i][j];
						}
					}
				}
			}
		}
	}
	
	public void fullConnectPoolMap() {
		double alpha = 1;
		int i=0;
		for(int n=0;n<poolLayer2Num;n++) {
			for(int h=0;h<poolLayer2MapHeight;h++) {
				for(int w=0;w<poolLayer2MapWidth;w++) {
					fullConnect[i] = alpha * poolLayer2Map[n][h][w];
					i++;
				}
			}
		}
	}
	
	public int softmaxFullConnect(int index) {
		unit = new double[unitNum];
		output = new double[outputNum];
		for(int i=0;i<unitNum;i++) {
			unit[i] = fullConnect[i] * fullConnectToUnitWeight[i];
		}
		double sum = 0.0;
		for(int j=0;j<outputNum;j++) {
			for(int i=0;i<unitNum;i++) {
				output[j] += unit[i] * unitToOutputWeight[i][j];
			}
			output[j] = Math.exp(output[j]);
			sum += output[j];
		}
		for(int j=0;j<outputNum;j++) {
			output[j] = output[j] / sum;
		}
		int maxIndex = 0;
		for(int i=0;i<outputNum;i++) {
			if(output[i]>output[maxIndex]) {
				maxIndex = i;
			}
		}
		if(maxIndex==index) {
			return 1;
		}else {
			return -1;
		}
	}
	
	public void backPropagation() {
		double alpha = 1;
		double[] outputVariance = new double[outputNum];
		for(int i=0;i<outputNum;i++) {
			outputVariance[i] = value[i] - output[i];
		}
		for(int i=0;i<unitNum;i++) {
			for(int j=0;j<outputNum;j++) {
				unitToOutputWeight[i][j] += alpha * unit[i] * output[j] * outputVariance[j];
			}
		}
		double beta = 1;
		double[] unitVariance = new double[unitNum];
		for(int i=0;i<unitNum;i++) {
			for(int j=0;j<outputNum;j++) {
				unitVariance[i] += unitToOutputWeight[i][j] * outputVariance[j];
			}
			unitVariance[i] = unitVariance[i] / outputNum;
		}
		for(int i=0;i<fullNum;i++) {
			fullConnectToUnitWeight[i] += beta * fullConnect[i] * 1 * unitVariance[i];
		}
		double[] fullVariance = new double[fullNum];
		for(int i=0;i<fullNum;i++) {
			fullVariance[i] = fullConnectToUnitWeight[i] * unitVariance[i];
		}
		double beta2 = 1;
		double[][][] poolLayer2Variance = new double[poolLayer2Num][poolLayer2MapHeight][poolLayer2MapWidth];
		int k=0;
		for(int n=0;n<poolLayer2Num;n++) {
			for(int i=0;i<poolLayer2MapHeight;i++) {
				for(int j=0;j<poolLayer2MapWidth;j++) {
					poolLayer2Variance[n][i][j] = fullVariance[k];
					k++;
				}
			}
		}
		double[][][] convLayer2Variance = new double[convLayer2Num][convLayer2MapHeight][convLayer2MapWidth];
		System.out.println(poolLayer2Num);
		System.out.println(poolLayer2MapHeight);
		System.out.println(poolLayer2MapWidth);
		System.out.println(convLayer2Num);
		System.out.println(convLayer2MapHeight);
		System.out.println(convLayer2MapWidth);
		for(int n=0;n<convLayer2Num;n++) {
			for(int i=0;i<convLayer2MapHeight;i++) {
				for(int j=0;j<convLayer2MapWidth;j++) {
					if (i==72 && j==72)
						convLayer2Variance[n][i][j] = poolLayer2Variance[n][35][35];
					else if (i==72 && j!=72)
						{convLayer2Variance[n][i][j] = poolLayer2Variance[n][35][j/2];}
					else if (i!=72 && j==72)
						{convLayer2Variance[n][i][j] = poolLayer2Variance[n][i/2][35];}
					else
						{convLayer2Variance[n][i][j] = poolLayer2Variance[n][i/2][j/2];}
				}
			}
		}
		for(int h=0;h<convLayer2MapHeight;h++) {
			for(int w=0;w<convLayer2MapWidth;w++) {
				for(int j=0;j<convLayer2Num;j++) {
					for(int i=0;i<poolLayer1Num;i++) {
						poolLayer1ToConvWeight[i][j][h][w] += beta2 * convLayer2Map[i][j][h][w] * 1 * convLayer2Variance[j][h][w];
					}
				}
			}
		}
	}
	
	public static void main(String args[]) throws IOException {
		CNNTwoLayersPCAInput cnn = new CNNTwoLayersPCAInput();
		int circle = 10000;
		int malwareTrainingNum = 2000;
		int malwareTotalNum = 5745;
		int benignTrainingNum = 2000;
		int benignTotalNum = 5242;
		
		cnn.averageMatrix(malwareTrainingNum,benignTrainingNum);
		System.out.println("average matrix over");
		cnn.coVarianceMatrix(malwareTrainingNum,benignTrainingNum);
		System.out.println("co-variance matrix over");
		cnn.getEigen();
		
		cnn.randomConvCore();
		cnn.initPoolCore();
		cnn.initFullConnectToUnitWeight();
		cnn.randomUnitToOutputWeight();
		cnn.randomPoolLayer1ToConvWeight();

		double avgPixel = cnn.getAveragePixel(malwareTrainingNum,benignTrainingNum);
		for(int c=0;c<circle;c++) {
			File folder = new File(cnn.malwareFolderPath);
			File[] fileList1 = folder.listFiles();
			folder = new File(cnn.benignFolderPath);
			File[] fileList2 = folder.listFiles();
			for(int j=0;j<fileList1.length && j<fileList2.length && j<malwareTrainingNum && j<benignTrainingNum;j++) {
				String fileName1 = cnn.malwareFolderPath+fileList1[j].getName();
				cnn.initMatrixAndValue(fileName1,avgPixel,0);
				cnn.convolutionMatrix();
				cnn.poolingConvLayer1Map();
				cnn.convolutionLayer1();
				cnn.layer1ToLayer2Weight();
				cnn.poolingConvLayer2Map();
				cnn.fullConnectPoolMap();
				cnn.softmaxFullConnect(0);
				cnn.backPropagation();
				String fileName2 = cnn.benignFolderPath+fileList2[j].getName();
				cnn.initMatrixAndValue(fileName2,avgPixel,1);
				cnn.convolutionMatrix();
				cnn.poolingConvLayer1Map();
				cnn.convolutionLayer1();
				cnn.layer1ToLayer2Weight();
				cnn.poolingConvLayer2Map();
				cnn.fullConnectPoolMap();
				cnn.softmaxFullConnect(1);
				cnn.backPropagation();
			}
			int right = 0;
			int wrong = 0;
			folder = new File(cnn.malwareFolderPath);
			fileList1 = folder.listFiles();
			for(int j=malwareTrainingNum;j<fileList1.length && j<malwareTotalNum;j++) {
				String fileName = cnn.malwareFolderPath+fileList1[j].getName();
				cnn.initMatrixAndValue(fileName,avgPixel,0);
				cnn.convolutionMatrix();
				cnn.poolingConvLayer1Map();
				cnn.convolutionLayer1();
				cnn.layer1ToLayer2Weight();
				cnn.poolingConvLayer2Map();
				cnn.fullConnectPoolMap();
				int rs = cnn.softmaxFullConnect(0);
				if(rs==1) {
					right++;
				}else {
					wrong++;
				}
			}
			System.out.println(c+"-----------------------------------------");
			double accuracy1 = (double)right * 100 / (right+wrong);
			System.out.println(right+" "+wrong+" "+accuracy1);
			right = 0;
			wrong = 0;
			folder = new File(cnn.malwareFolderPath);
			fileList1 = folder.listFiles();
			for(int j=0;j<fileList1.length && j<malwareTrainingNum;j++) {
				String fileName = cnn.malwareFolderPath+fileList1[j].getName();
				cnn.initMatrixAndValue(fileName,avgPixel,0);
				cnn.convolutionMatrix();
				cnn.poolingConvLayer1Map();
				cnn.convolutionLayer1();
				cnn.layer1ToLayer2Weight();
				cnn.poolingConvLayer2Map();
				cnn.fullConnectPoolMap();
				int rs = cnn.softmaxFullConnect(0);
				if(rs==1) {
					right++;
				}else {
					wrong++;
				}
			}
			double accuracy2 = (double)right * 100 / (right+wrong);
			System.out.println(right+" "+wrong+" "+accuracy2);
			right = 0;
			wrong = 0;
			folder = new File(cnn.benignFolderPath);
			fileList2 = folder.listFiles();
			for(int j=benignTrainingNum;j<fileList2.length && j<benignTotalNum;j++) {
				String fileName = cnn.benignFolderPath+fileList2[j].getName();
				cnn.initMatrixAndValue(fileName,avgPixel,1);
				cnn.convolutionMatrix();
				cnn.poolingConvLayer1Map();
				cnn.convolutionLayer1();
				cnn.layer1ToLayer2Weight();
				cnn.poolingConvLayer2Map();
				cnn.fullConnectPoolMap();
				int rs = cnn.softmaxFullConnect(1);
				if(rs==1) {
					right++;
				}else {
					wrong++;
				}
			}
			double accuracy3 = (double)right * 100 / (right+wrong);
			System.out.println(right+" "+wrong+" "+accuracy3);
			right = 0;
			wrong = 0;
			folder = new File(cnn.benignFolderPath);
			fileList2 = folder.listFiles();
			for(int j=0;j<fileList2.length && j<benignTrainingNum;j++) {
				String fileName = cnn.benignFolderPath+fileList2[j].getName();
				cnn.initMatrixAndValue(fileName,avgPixel,1);
				cnn.convolutionMatrix();
				cnn.poolingConvLayer1Map();
				cnn.convolutionLayer1();
				cnn.layer1ToLayer2Weight();
				cnn.poolingConvLayer2Map();
				cnn.fullConnectPoolMap();
				int rs = cnn.softmaxFullConnect(1);
				if(rs==1) {
					right++;
				}else {
					wrong++;
				}
			}
			double accuracy4 = (double)right * 100 / (right+wrong);
			System.out.println(right+" "+wrong+" "+accuracy4);
			double accuracy = (accuracy1 + accuracy3) / 2;
			System.out.println("Detection accuracy: "+accuracy+" %");
			accuracy = (accuracy2 + accuracy4) / 2;
			System.out.println("Training accuracy: "+accuracy+" %");
		}
	}

}
