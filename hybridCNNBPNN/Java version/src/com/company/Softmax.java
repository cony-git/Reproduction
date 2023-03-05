package com.company;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;


public class Softmax {
	
	int unitNum = 7220+100;
	int outputNum = 2;
	double[] unit = new double[unitNum];
	double[] output = new double[outputNum];
	double[][] unitToOutputWeight = new double[unitNum][outputNum];
	double[] value = new double[outputNum];
	
	public void randomUnitToOutputWeight() {
		Random rand = new Random();
		for(int i=0;i<unitNum;i++) {
			for(int j=0;j<outputNum;j++) {
				double temp = rand.nextInt(1000);
				temp = temp / 1000;
				unitToOutputWeight[i][j]=1.0;//temp;
			}
		}
	}
	
	public int softmaxFullConnect(int index) {
		output = new double[outputNum];
		double sum = 0.0;
		for(int j=0;j<outputNum;j++) {
			for(int i=0;i<unitNum;i++) {
				output[j] += unit[i] * unitToOutputWeight[i][j];
			}
			output[j] = Math.exp(output[j]);
			sum += output[j];
		}
		if(sum!=0) {
			for(int j=0;j<outputNum;j++) {
				output[j] = output[j] / sum;
			}
		}else {
			return 0;
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
	}
	
	public static void main(String args[]) {
		try {
			Softmax softmax = new Softmax();
			softmax.randomUnitToOutputWeight();
			List<double[]> malwareList = new ArrayList<double[]>();
			List<double[]> benignList = new ArrayList<double[]>();
			Scanner in = new Scanner(new File("E:\\Dataset(A feature-hybrid malware variants detection )\\malwareAPIFeature.txt"));
			while(in.hasNext()) {
				String line = in.nextLine();
				String[] temp = line.split(" ");
				double[] u = new double[softmax.unitNum];
				for(int i=0;i<temp.length;i++) {
//					u[i] = Double.parseDouble(temp[i]);
				}
				malwareList.add(u);
			}
			int n = 0;
			in = new Scanner(new File("E:\\Dataset(A feature-hybrid malware variants detection )\\malwareOpcodeFeature.txt"));
			while(in.hasNext()) {
				String line = in.nextLine();
				String[] temp = line.split(" ");
				for(int i=0;i<temp.length;i++) {
					malwareList.get(n)[i+100] = Double.parseDouble(temp[i]);
				}
				n++;
			}
			in = new Scanner(new File("E:\\Dataset(A feature-hybrid malware variants detection )\\benignAPIFeature.txt"));
			while(in.hasNext()) {
				String line = in.nextLine();
				String[] temp = line.split(" ");
				double[] u = new double[softmax.unitNum];
				for(int i=0;i<temp.length;i++) {
//					u[i] = Double.parseDouble(temp[i]);
				}
				benignList.add(u);
			}
			n = 0;
			in = new Scanner(new File("E:\\Dataset(A feature-hybrid malware variants detection )\\benignOpcodeFeature.txt"));
			while(in.hasNext()) {
				String line = in.nextLine();
				String[] temp = line.split(" ");
				for(int i=0;i<temp.length;i++) {
					benignList.get(n)[i+100] = Double.parseDouble(temp[i]);
				}
				n++;
			}
			for(int c=0;c<1000;c++) {
				for(int i=0;i<350;i++) {
					softmax.unit = malwareList.get(i);
					softmax.softmaxFullConnect(0);
					softmax.value[0] = 1.0;
					softmax.value[1] = 0.0;
					softmax.backPropagation();

					
					softmax.unit = benignList.get(i);
					softmax.softmaxFullConnect(1);
					softmax.value[0] = 0.0;
					softmax.value[1] = 1.0;
					softmax.backPropagation();
				}
				double right = 0;
				double wrong = 0;
				for(int i=350;i<1000;i++) {
					softmax.unit = malwareList.get(i);
					int rs = softmax.softmaxFullConnect(0);
					if(rs==1) {
						right++;
					}else if(rs==-1) {
						wrong++;
					}
					softmax.unit = benignList.get(i);
					rs = softmax.softmaxFullConnect(1);
					if(rs==1) {
						right++;
					}else if(rs==-1) {
						wrong++;
					}
				}
				System.out.println(right+" "+wrong);
				System.out.println(right/(right+wrong));
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}

}
