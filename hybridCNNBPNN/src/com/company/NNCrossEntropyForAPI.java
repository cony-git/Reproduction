package com.company;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.Scanner;


public class NNCrossEntropyForAPI {
	
	public int inputNumber = 10;
	public int outputNumber = 2;
	public int unitNumber = 100;
	public int circle = 100000;
	public int trainNumber = 1000;
	public int detectNumber = 1000;
	public List<String> systemCallList = new ArrayList<String>(); //APIList
	public List<double[]> malwareSystemCallProfiles = new ArrayList<double[]>();
	public List<double[]> benignSystemCallProfiles = new ArrayList<double[]>();
	
	public double[] unit;
	public double[] output;
	public double[] value;
	public double[][] inputToUnitWeight;
	public double[][] unitToOutputWeight;
	
	public void getAPICall() {
		try {
			try {
				Scanner in = new Scanner(new File("C:\\Users\\Cony\\Desktop\\hybridCNNBPNN\\src\\com\\company\\API.txt"));
				while(in.hasNext()) {
					String line = in.nextLine();
					line = line.toLowerCase();
					systemCallList.add(line);
				}
				File folder0 = new File("E:\\Dataset(A feature-hybrid malware variants detection )\\malicious");
				File[] fileList0 = folder0.listFiles();
				for(int i=0;i<fileList0.length;i++) {
					System.out.println(i);
					int total = 0;
					double[] temp = new double[systemCallList.size()];
					in = new Scanner(fileList0[i]);
					while(in.hasNext()) {
						String line = in.nextLine();
						line = line.toLowerCase();
						for(int n=0;n<systemCallList.size();n++) {
							if(line.contains(systemCallList.get(n))) {
								temp[n]++;
								total++;
							}
						}
					}
					if(total!=0) {
						for(int h=0;h<systemCallList.size();h++) {
							temp[h] = temp[h] / total;
						}
					}
					malwareSystemCallProfiles.add(temp);
				}
				FileWriter fw = null;
				File f=new File("E:\\Dataset(A feature-hybrid malware variants detection )\\malwareAPIList.txt");
				fw = new FileWriter(f, true);
				PrintWriter pw = new PrintWriter(fw);
				for(int h=0;h<malwareSystemCallProfiles.size();h++) {
					for(int w=0;w<systemCallList.size();w++) {
						pw.print(malwareSystemCallProfiles.get(h)[w]+" ");
						pw.flush();
					}
					pw.println();
					pw.flush();
				}
				fw.flush();
				pw.close();
				fw.close();
				folder0 = new File("E:\\Dataset(A feature-hybrid malware variants detection )\\benign");
				fileList0 = folder0.listFiles();
				for(int i=0;i<fileList0.length;i++) {
					System.out.println(i);
					int total = 0;
					double[] temp = new double[systemCallList.size()];
					in = new Scanner(fileList0[i]);
					while(in.hasNext()) {
						String line = in.nextLine();
						line = line.toLowerCase();
						for(int n=0;n<systemCallList.size();n++) {
							if(line.contains(systemCallList.get(n))) {
								temp[n]++;
								total++;
							}
						}
					}
					if(total!=0) {
						for(int h=0;h<systemCallList.size();h++) {
							temp[h] = temp[h] / total;
						}
					}
					benignSystemCallProfiles.add(temp);
				}
				fw = null;
				f=new File("E:\\Dataset(A feature-hybrid malware variants detection )\\benignAPIList.txt");
				fw = new FileWriter(f, true);
				pw = new PrintWriter(fw);
				for(int h=0;h<benignSystemCallProfiles.size();h++) {
					for(int w=0;w<systemCallList.size();w++) {
						pw.print(benignSystemCallProfiles.get(h)[w]+" ");
						pw.flush();
					}
					pw.println();
					pw.flush();
				}
			}catch(Exception e) {
				e.printStackTrace();
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void getAPICallList() {
		try {
			try {
				Scanner in = new Scanner(new File("lib\\API.txt"));
				while(in.hasNext()) {
					String line = in.nextLine();
					line = line.toLowerCase();
					systemCallList.add(line);
				}
				File file = new File("D:\\malwareAPIList.txt");
				in = new Scanner(file);
				while(in.hasNext()) {
					double[] temp = new double[systemCallList.size()];
					String line = in.nextLine();
					String[] API = line.split(" ");
					double total = 0;
					for(int h=0;h<systemCallList.size();h++) {
						temp[h] = Double.parseDouble(API[h]);
						total += temp[h];
					}
//					System.out.println(total);
					malwareSystemCallProfiles.add(temp);
				}
				file = new File("D:\\benignAPIList.txt");
				in = new Scanner(file);
				while(in.hasNext()) {
					double[] temp = new double[systemCallList.size()];
					String line = in.nextLine();
					String[] API = line.split(" ");
					double total = 0;
					for(int h=0;h<systemCallList.size();h++) {
						temp[h] = Double.parseDouble(API[h]);
						total += temp[h];
					}
//					System.out.println(total);
					benignSystemCallProfiles.add(temp);
				}
			}catch(Exception e) {
				e.printStackTrace();
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void init() {
		inputNumber = systemCallList.size();
		outputNumber = 2;
		unitNumber = 100;
		unit = new double[unitNumber];
		output = new double[outputNumber];
		value = new double[outputNumber];
		inputToUnitWeight = new double[inputNumber][unitNumber];
		unitToOutputWeight = new double[unitNumber][outputNumber];
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
		Random rand = new Random();
		for(int i=0;i<unitNumber;i++) {
			for(int j=0;j<outputNumber;j++) {
				double tempWeight = rand.nextInt(1000);
				tempWeight = tempWeight / 1000;
				unitToOutputWeight[i][j]=tempWeight;
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
			//System.out.println(output[j] + " ");
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
		double alpha = 1;
		double[] outputVariance = new double[outputNumber];
		for(int i=0;i<outputNumber;i++) {
			if(output[i]<0 && output[i]>1) {
				return;
			}
			outputVariance[i] = value[i] - output[i];
		}
		for(int i=0;i<unitNumber;i++) {
			for(int j=0;j<outputNumber;j++) {
				unitToOutputWeight[i][j] += alpha * unit[i] * outputVariance[j];
			}
		}
		double beta = 1;
		double[] unitVariance = new double[unitNumber];
		for(int i=0;i<unitNumber;i++) {
			for(int j=0;j<outputNumber;j++) {
				unitVariance[i] += unitToOutputWeight[i][j] * outputVariance[j];
			}
			unitVariance[i] = unitVariance[i] / outputNumber;
		}
		for(int i=0;i<inputNumber;i++) {
			for(int j=0;j<unitNumber;j++) {
				inputToUnitWeight[i][j] += beta * input[i] * unitVariance[j];
			}
		}
	}
	
	public static void main(String args[]) {
		NNCrossEntropyForAPI nn = new NNCrossEntropyForAPI();
		nn.getAPICall();
//		nn.getAPICallList();
		nn.init();
		nn.randomInputToUnitWeight();
		nn.initUnitToOutputWeight();
		for(int n=0;n<nn.circle;n++) {
			for(int i=0;i<130;i++) {
				nn.value[0] = 1.0;
				nn.value[1] = 0.0;
				nn.foreward(nn.malwareSystemCallProfiles.get(i),0);
				nn.backPropagation(nn.malwareSystemCallProfiles.get(i));
				nn.value[0] = 0.0;
				nn.value[1] = 1.0;
				nn.foreward(nn.benignSystemCallProfiles.get(i),1);
				nn.backPropagation(nn.benignSystemCallProfiles.get(i));
			}
			if(n % 1==0) {
				FileWriter fw1 = null;
				FileWriter fw2 = null;
				try {
					File f1=new File("E:\\Dataset(A feature-hybrid malware variants detection )\\malwareAPIFeature.txt");
					f1.delete();
					fw1 = new FileWriter(f1, true);
					File f2=new File("E:\\Dataset(A feature-hybrid malware variants detection )\\benignAPIFeature.txt");
					f2.delete();
					fw2 = new FileWriter(f2, true);
				} catch (Exception e) {
					e.printStackTrace();
				}
				PrintWriter pw1 = new PrintWriter(fw1);
				PrintWriter pw2 = new PrintWriter(fw2);
				System.out.println("-----------------------------------------------------------");
				double right = 0.0;
				double wrong = 0.0;
				nn.value[0] = 1.0;
				nn.value[1] = 0.0;
				for(int i=0;i<nn.malwareSystemCallProfiles.size() && i<nn.trainNumber;i++) {
					int temp = nn.foreward(nn.malwareSystemCallProfiles.get(i),0);
					for(int j=0;j<nn.unitNumber;j++) {
						pw1.print(nn.unit[j]+" ");
					}
					pw1.println();
					pw1.flush();
					if(temp == 1) {
						right++;
					}else if(temp == -1) {
						wrong++;
					}
				}
				pw1.close();
				double right2 = 0.0;
				double wrong2 = 0.0;
				nn.value[0] = 0.0;
				nn.value[1] = 1.0;
				for(int i=0;i<nn.benignSystemCallProfiles.size() && i<nn.trainNumber;i++) {
					int temp = nn.foreward(nn.benignSystemCallProfiles.get(i),1);
					for(int j=0;j<nn.unitNumber;j++) {
						pw2.print(nn.unit[j]+" ");
					}
					pw2.println();
					pw2.flush();
					if(temp == 1) {
						right2++;
					}else if(temp == -1) {
						wrong2++;
					}
				}
				pw2.close();
				Date d = new Date();
				System.out.println("Date Time: " + d);
				System.out.println("circle: " + n);
				System.out.println("right: " + right+" wrong: "+wrong);
				System.out.println("right: " + right2+" wrong: "+wrong2);
				System.out.println("Accuracy: " + (right * 100 / (right + wrong)) + " %");
				System.out.println("Accuracy: " + (right2 * 100 / (right2 + wrong2)) + " %");
//				right = 0.0;
//				wrong = 0.0;
//				nn.value[0] = 1.0;
//				nn.value[1] = 0.0;
//				for(int i=nn.trainNumber;i<nn.malwareSystemCallProfiles.size() && i<nn.detectNumber;i++) {
//					int temp = nn.foreward(nn.malwareSystemCallProfiles.get(i),0);
//					if(temp == 1) {
//						right++;
//					}else if(temp == -1) {
//						wrong++;
//					}
//				}
//				right2 = 0.0;
//				wrong2 = 0.0;
//				nn.value[0] = 0.0;
//				nn.value[1] = 1.0;
//				for(int i=nn.trainNumber;i<nn.benignSystemCallProfiles.size() && i<nn.detectNumber;i++) {
//					int temp = nn.foreward(nn.benignSystemCallProfiles.get(i),1);
//					if(temp == 1) {
//						right2++;
//					}else if(temp == -1) {
//						wrong2++;
//					}
//				}
//				d = new Date();
//				System.out.println("Date Time: " + d);
//				System.out.println("circle: " + n);
//				System.out.println("right: " + right+" wrong: "+wrong);
//				System.out.println("right: " + right2+" wrong: "+wrong2);
//				System.out.println("Accuracy: " + (right * 100 / (right + wrong)) + " %");
//				System.out.println("Accuracy: " + (right2 * 100 / (right2 + wrong2)) + " %");
//				FileWriter fw = null;
//				try {
//					File f=new File("D:\\DNN1Result.txt");
//					fw = new FileWriter(f, true);
//				} catch (Exception e) {
//					e.printStackTrace();
//				}
//				PrintWriter pw = new PrintWriter(fw);
//				pw.println("Date Time: " + d);
//				pw.println("circle: " + n);
//				pw.println("right: " + right+" wrong: "+wrong);
//				pw.println("right: " + right2+" wrong: "+wrong2);
//				pw.println("Accuracy: " + (right * 100 / (right + wrong)) + " %");
//				pw.println("Accuracy: " + (right2 * 100 / (right2 + wrong2)) + " %");
//				pw.flush();
//				try {
//					fw.flush();
//					pw.close();
//					fw.close();
//				} catch (Exception e) {
//					e.printStackTrace();
//				}
			}
		}
	}
}
