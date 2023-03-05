package com.company;

import Jama.Matrix;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import javax.imageio.ImageIO;


public class PCAInitializedNN {
	
	public int pcaDim = 200;
	public double[] avgInput = null;
	public double[][] coVarianceMatrix = null;
	public double[][] eigenVectors = null;
	
	public int inputNumber = pcaDim;
	public int outputNumber = 2;
	public int unitNumber = 50;
	public int circle = 50000;
	public int trainNumber = 2000;
	public int detectNumber = 2000;
	public List<String> APIList = new ArrayList<String>();
	public List<double[]> malwareAPIProfiles = new ArrayList<double[]>();
	public List<double[]> benignAPIProfiles = new ArrayList<double[]>();
	public List<double[]> malwareAPIPCAInitializedProfiles = new ArrayList<double[]>();
	public List<double[]> benignAPIPCAInitializedProfiles = new ArrayList<double[]>();
	
	public double[] unit;
	public double[] output;
	public double[] value;
	public double[][] inputToUnitWeight;
	public double[][] unitToOutputWeight;
	
	public String malwareFolderPath = "";
	public String benignFolderPath = "";
	
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
	
	public void initPCA() {
		Scanner in;
		try {
			in = new Scanner(new File("data/APIList.txt"));
			while (in.hasNextLine()) {
				String API = in.nextLine();
				APIList.add(API);
	        }
			in.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		inputNumber = APIList.size();
		File folder0 = new File(malwareFolderPath);
		File[] fileList0 = folder0.listFiles();
		for(int i=0;i<fileList0.length && i<trainNumber;i++) {
			if(fileList0[i].getName().contains(".txt")==false || fileList0[i].getName().contains("Trojan")==false) {
				continue;
			}
			double[] input = new double[inputNumber];
			int totalCount = 0;
			for(int j=0;j<inputNumber;j++) {
				input[j] = 0.0;
			}
			try {
				in = new Scanner(fileList0[i]);
				while (in.hasNextLine()) {
					String tempStr = in.nextLine();
					int index = tempStr.lastIndexOf("@");
					String API = tempStr.substring(index+1);
					for(int j=0;j<APIList.size();j++) {
						if(API.equals(APIList.get(j))==true) {
							input[j]++;
							totalCount++;
							break;
						}
					}
		        }
				in.close();
				if(totalCount>0) {
					for(int j=0;j<inputNumber;j++) {
						input[j] = input[j] / totalCount;
					}
					malwareAPIProfiles.add(input);
				}
			}catch(Exception e) {
				e.printStackTrace();
			}
		}
		File folder1 = new File(benignFolderPath);
		File[] fileList1 = folder1.listFiles();
		for(int i=0;i<fileList1.length && i<trainNumber;i++) {
			if(fileList1[i].getName().contains(".txt")==false) {
				continue;
			}
			double[] input = new double[inputNumber];
			int totalCount = 0;
			for(int j=0;j<inputNumber;j++) {
				input[j] = 0.0;
			}
			try {
				in = new Scanner(fileList1[i]);
				while (in.hasNextLine()) {
					String tempStr = in.nextLine();
					int index = tempStr.lastIndexOf("@");
					String API = tempStr.substring(index+1);
					for(int j=0;j<APIList.size();j++) {
						if(API.equals(APIList.get(j))==true) {
							input[j]++;
							totalCount++;
							break;
						}
					}
		        }
				in.close();
				if(totalCount>0) {
					for(int j=0;j<inputNumber;j++) {
						input[j] = input[j] / totalCount;
					}
					benignAPIProfiles.add(input);
				}
			}catch(Exception e) {
				e.printStackTrace();
			}
		}
		
		avgInput = new double[inputNumber];
		coVarianceMatrix = new double[inputNumber][inputNumber];
		eigenVectors = new double[inputNumber][pcaDim];
		for(int i=0;i<malwareAPIProfiles.size();i++) {
			for(int k=0;k<inputNumber;k++) {
				avgInput[k] += malwareAPIProfiles.get(i)[k];
			}
		}
		for(int i=0;i<benignAPIProfiles.size();i++) {
			for(int k=0;k<inputNumber;k++) {
				avgInput[k] += benignAPIProfiles.get(i)[k];
			}
		}
		for(int k=0;k<inputNumber;k++) {
			avgInput[k] = avgInput[k] / (malwareAPIProfiles.size()+benignAPIProfiles.size());
		}
		
		outputNumber = 2;
		unitNumber = 50;
		unit = new double[unitNumber];
		output = new double[outputNumber];
		value = new double[outputNumber];
		inputToUnitWeight = new double[pcaDim][unitNumber];
		unitToOutputWeight = new double[unitNumber][outputNumber];
	}
	
	public void coVarianceMatrix() {
		try {
			for(int i=0;i<malwareAPIProfiles.size();i++) {
				double[] trainInput = new double[inputNumber];
				for(int k=0;k<inputNumber;k++) {
					trainInput[k] = malwareAPIProfiles.get(i)[k];
				}
				double[][] variance = new double[1][inputNumber];
				for(int k=0;k<inputNumber;k++) {
					variance[0][k] = trainInput[k] - avgInput[k];
				}
				double[][] tempCoVariance = multiplyMatrix(transpositionMatrix(variance,1,inputNumber),inputNumber,1,variance,1,inputNumber);
				for(int m=0;m<inputNumber;m++) {
					for(int n=0;n<inputNumber;n++) {
						coVarianceMatrix[m][n] += tempCoVariance[m][n];
					}
				}
			}
			for(int i=0;i<benignAPIProfiles.size();i++) {
				double[] trainInput = new double[inputNumber];
				for(int k=0;k<inputNumber;k++) {
					trainInput[k] = benignAPIProfiles.get(i)[k];
				}
				double[][] variance = new double[1][inputNumber];
				for(int k=0;k<inputNumber;k++) {
					variance[0][k] = trainInput[k] - avgInput[k];
				}
				double[][] tempCoVariance = multiplyMatrix(transpositionMatrix(variance,1,inputNumber),inputNumber,1,variance,1,inputNumber);
				for(int m=0;m<inputNumber;m++) {
					for(int n=0;n<inputNumber;n++) {
						coVarianceMatrix[m][n] += tempCoVariance[m][n];
					}
				}
			}
			for(int m=0;m<inputNumber;m++) {
				for(int n=0;n<inputNumber;n++) {
					coVarianceMatrix[m][n] = coVarianceMatrix[m][n] / (malwareAPIProfiles.size()+benignAPIProfiles.size());
				}
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}

	public void getEigen() {
		try {
			Matrix e = new Matrix(coVarianceMatrix);
			for(int i=0;i<inputNumber;i++) {
				for(int j=0;j<inputNumber;j++) {
					eigenVectors[i][j]=e.eig().getV().get(i,j);}}
			for(int i=0;i<inputNumber;i++) {
				for(int j=inputNumber-pcaDim;j<inputNumber;j++) {
					eigenVectors[i][j-(inputNumber-pcaDim)] = eigenVectors[i][j];
				}
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void initNN() {
		for(int i=0;i<malwareAPIProfiles.size();i++) {
			double[] trainInput = new double[inputNumber];
			for(int k=0;k<inputNumber;k++) {
				trainInput[k] = malwareAPIProfiles.get(i)[k];
			}
			double[][] variance = new double[1][inputNumber];
			for(int k=0;k<inputNumber;k++) {
				variance[0][k] = trainInput[k] - avgInput[k];
			}
			double[][] eigenMatrix = multiplyMatrix(variance,1,inputNumber,eigenVectors,inputNumber,pcaDim);
			double[] input = new double[pcaDim];
			for(int k=0;k<pcaDim;k++) {
				input[k] = eigenMatrix[0][k];
			}
			malwareAPIPCAInitializedProfiles.add(input);
		}
		for(int i=0;i<benignAPIProfiles.size();i++) {
			double[] trainInput = new double[inputNumber];
			for(int k=0;k<inputNumber;k++) {
				trainInput[k] = benignAPIProfiles.get(i)[k];
			}
			double[][] variance = new double[1][inputNumber];
			for(int k=0;k<inputNumber;k++) {
				variance[0][k] = trainInput[k] - avgInput[k];
			}
			double[][] eigenMatrix = multiplyMatrix(variance,1,inputNumber,eigenVectors,inputNumber,pcaDim);
			double[] input = new double[pcaDim];
			for(int k=0;k<pcaDim;k++) {
				input[k] = eigenMatrix[0][k];
			}
			benignAPIPCAInitializedProfiles.add(input);
		}
		inputNumber = pcaDim;
	}
	
	public void randomInputToUnitWeight() {
		Random rand = new Random();
		for(int i=0;i<inputNumber;i++) {
			for(int j=0;j<unitNumber;j++) {
				double tempWeight = rand.nextInt(1000);
				tempWeight = tempWeight / 1000;
				inputToUnitWeight[i][j]=tempWeight;
			}
		}
	}
	
	public void initUnitToOutputWeight() {
		for(int i=0;i<unitNumber;i++) {
			for(int j=0;j<outputNumber;j++) {
				unitToOutputWeight[i][j]=1.0;
			}
		}
	}
	
	public int foreward(double[] input, int index) {
		unit = new double[unitNumber];
		output = new double[outputNumber];
		for(int j=0;j<unitNumber;j++) {
			for(int i=0;i<inputNumber;i++) {
				unit[j] += input[i] * inputToUnitWeight[i][j];
			}
			unit[j] = 1 / (1 + Math.exp(0 - unit[j]));
		}
		for(int j=0;j<outputNumber;j++) {
			for(int i=0;i<unitNumber;i++) {
				output[j] += unit[i] * unitToOutputWeight[i][j];
			}
			output[j] = 1 / (1 + Math.exp(0 - output[j]));
		}
		int maxIndex = 0;
		for(int i=1;i<outputNumber;i++) {
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
	
	public void backPropagation(double[] input) {
		double alpha = 0.5;
		double[] outputVariance = new double[outputNumber];
		for(int i=0;i<outputNumber;i++) {
			outputVariance[i] = value[i] - output[i];
		}
		for(int i=0;i<unitNumber;i++) {
			for(int j=0;j<outputNumber;j++) {
				unitToOutputWeight[i][j] += alpha * unit[i] * (output[j] * (1 - output[j])) * outputVariance[j];
			}
		}
		double beta = 0.5;
		double[] unitVariance = new double[unitNumber];
		for(int i=0;i<unitNumber;i++) {
			for(int j=0;j<outputNumber;j++) {
				unitVariance[i] += unitToOutputWeight[i][j] * outputVariance[j];
			}
			unitVariance[i] = unitVariance[i] / outputNumber;
		}
		for(int i=0;i<inputNumber;i++) {
			for(int j=0;j<unitNumber;j++) {
				inputToUnitWeight[i][j] += beta * input[i] * (unit[j] * (1 - unit[j])) * unitVariance[j];
			}
		}
	}
	
	public void histogramForAPIs() {
		double[] avgInput = new double[inputNumber];
		double[] avgMalwareInput = new double[inputNumber];
		for(int i=0;i<malwareAPIProfiles.size();i++) {
			for(int k=0;k<inputNumber;k++) {
				avgMalwareInput[k] += malwareAPIProfiles.get(i)[k];
			}
		}
		double[] avgBenignInput = new double[inputNumber];
		for(int i=0;i<benignAPIProfiles.size();i++) {
			for(int k=0;k<inputNumber;k++) {
				avgBenignInput[k] += benignAPIProfiles.get(i)[k];
				if(i<benignAPIProfiles.size()/2) {
					avgInput[k] += benignAPIProfiles.get(i)[k];
				}
			}
		}
		System.out.println("-----------------------------------------");
		for(int k=0;k<inputNumber;k++) {
			avgMalwareInput[k] = avgMalwareInput[k] / malwareAPIProfiles.size();
			System.out.println(avgMalwareInput[k]*100);
		}
		System.out.println("-----------------------------------------");
		for(int k=0;k<inputNumber;k++) {
			avgBenignInput[k] = avgBenignInput[k] / benignAPIProfiles.size();
			System.out.println(avgBenignInput[k]*100);
		}
		System.out.println("-----------------------------------------");
		for(int k=0;k<inputNumber;k++) {
			avgInput[k] = avgInput[k] / (benignAPIProfiles.size()/2);
			System.out.println(avgInput[k]*100);
		}
		System.out.println("-----------------------------------------");
		for(int k=0;k<inputNumber;k++) {
			if(avgMalwareInput[k]/avgBenignInput[k]>1 && avgMalwareInput[k]>0.01) {
				System.out.println(APIList.get(k));
			}
		}
		System.out.println("---------------------------------size:"+inputNumber);
	}
	
	public static void main(String args[]) {
		PCAInitializedNN nn = new PCAInitializedNN();
		nn.initPCA();
		nn.histogramForAPIs();
		nn.coVarianceMatrix();
		nn.getEigen();
		nn.initNN();
		nn.randomInputToUnitWeight();
		nn.initUnitToOutputWeight();
		
		for(int n=0;n<nn.circle;n++) {
			nn.value[0] = 1.0;
			nn.value[1] = 0.0;
			for(int i=0;i<nn.malwareAPIPCAInitializedProfiles.size() && i<nn.trainNumber;i++) {
				nn.foreward(nn.malwareAPIPCAInitializedProfiles.get(i),0);
				nn.backPropagation(nn.malwareAPIPCAInitializedProfiles.get(i));
			}
			nn.value[0] = 0.0;
			nn.value[1] = 1.0;
			for(int i=0;i<nn.benignAPIPCAInitializedProfiles.size() && i<nn.trainNumber;i++) {
				nn.foreward(nn.benignAPIPCAInitializedProfiles.get(i),1);
				nn.backPropagation(nn.benignAPIPCAInitializedProfiles.get(i));
			}
			if(n % 10==0) {
				double right = 0.0;
				double wrong = 0.0;
				for(int i=0;i<nn.malwareAPIPCAInitializedProfiles.size() && i<nn.detectNumber;i++) {
					int temp = nn.foreward(nn.malwareAPIPCAInitializedProfiles.get(i),0);
					if(temp == 1) {
						right++;
					}else if(temp == -1) {
						wrong++;
					}
				}
				double right2 = 0.0;
				double wrong2 = 0.0;
				for(int i=0;i<nn.benignAPIPCAInitializedProfiles.size() && i<nn.detectNumber;i++) {
					int temp = nn.foreward(nn.benignAPIPCAInitializedProfiles.get(i),1);
					if(temp == 1) {
						right2++;
					}else if(temp == -1) {
						wrong2++;
					}
				}
				Date d = new Date();
				System.out.println("Date Time: " + d);
				System.out.println("circle: " + n);
				System.out.println("right: " + right+" wrong: "+wrong);
				System.out.println("right: " + right2+" wrong: "+wrong2);
				System.out.println("Accuracy: " + (right * 100 / (right + wrong)) + " %");
				System.out.println("Accuracy: " + (right2 * 100 / (right2 + wrong2)) + " %");
				
			}
		}
	}
}
