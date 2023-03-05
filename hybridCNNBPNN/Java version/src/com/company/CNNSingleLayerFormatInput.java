package com.company;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import javax.imageio.ImageIO;

public class CNNSingleLayerFormatInput {
	
	public int matrixWidth = 159;
	public int matrixHeight = 159;
	public int convSize = 5;
	public int convMapWidth = matrixWidth - convSize + 1;
	public int convMapHeight = matrixHeight - convSize + 1;
	public int poolSize = 4;
	public int poolMapWidth = convMapWidth / poolSize;
	public int poolMapHeight = convMapHeight / poolSize;
	public int convNum = 5;
	public int poolNum = convNum;
	public int fullNum = poolNum * poolMapWidth * poolMapHeight;
	public int unitNum = fullNum;
	public int outputNum = 2;
	public int valueNum = outputNum;
	
	public double[][] matrix = new double[matrixHeight][matrixWidth];
	public double[][][] convCore = new double[convNum][convSize][convSize];
	public double[][][] convMap = new double[convNum][convMapHeight][convMapWidth];
	public double[][][] poolCore = new double[poolNum][poolSize][poolSize];
	public double[][][] poolMap = new double[poolNum][poolMapHeight][poolMapWidth];
	public double[] fullConnect = new double[fullNum];
	public double[] unit = new double[unitNum];
	public double[] output = new double[outputNum];
	public double[] value = new double[valueNum];
	
	//public double[][][] convMapWeight = new double[convNum][convMapHeight][convMapWidth];
	public double[] fullConnectToUnitWeight = new double[fullNum];
	public double[][] unitToOutputWeight = new double[unitNum][outputNum];
	
	public void randomConvCore() {
		Random rand = new Random();
		for(int n=0;n<convNum;n++) {
			for(int i=0;i<convSize;i++) {
				for(int j=0;j<convSize;j++) {
					double temp = rand.nextInt(1000);
					temp = temp / 1000;
					convCore[n][i][j]=temp;
				}
			}
		}
	}
	
	public void initPoolCore() {
		for(int n=0;n<poolNum;n++) {
			for(int i=0;i<poolSize;i++) {
				for(int j=0;j<poolSize;j++) {
					poolCore[n][i][j]=1.0 / (poolSize * poolSize);
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
	
	public void initMatrixAndValue(String filePathName,int index) throws IOException {
		for(int i=0;i<valueNum;i++) {
			if(i==index) {
				value[i] = 1;
			}else {
				value[i] = 0;
			}
		}
		File file = new File(filePathName);
		BufferedImage image = ImageIO.read(file);
		int width = image.getWidth();
		matrixWidth = width;
		int height = image.getHeight();
		matrixHeight = height;
		for(int i=0;i<width;i++) {
			for(int j=0;j<height;j++) {
				int pixel = image.getRGB(i, j);
				int[] rgb = new int[3];
				rgb[0] = (pixel & 0xff0000) >> 16;
				matrix[j][i] = rgb[0];
			}
		}
		for(int i=0;i<width;i++) {
			for(int j=0;j<height;j++) {
//				matrix[j][i] = matrix[j][i] - averagePixel;
				matrix[j][i] = matrix[j][i] / (255 * matrixHeight * matrixWidth);
			}
		}
	}
	
	public void convolutionMatrix() {
		for(int n=0;n<convNum;n++) {
			for(int h=0;h<convMapHeight;h++) {
				for(int w=0;w<convMapWidth;w++) {
					convMap[n][h][w] = 0;
					for(int i=0;i<convSize;i++) {
						for(int j=0;j<convSize;j++) {
							convMap[n][h][w] += matrix[h+i][w+j] * convCore[n][i][j];
						}
					}
					convMap[n][h][w] = convMap[n][h][w] / (convSize * convSize);
				}
			}
		}
	}
	
	public void poolingConvMap() {
		for(int n=0;n<poolNum;n++) {
			for(int h=0;h<poolMapHeight;h++) {
				for(int w=0;w<poolMapWidth;w++) {
					poolMap[n][h][w] = 0;
					for(int i=0;i<poolSize;i++) {
						for(int j=0;j<poolSize;j++) {
							poolMap[n][h][w] += convMap[n][2*h+i][2*w+j] * poolCore[n][i][j];
						}
					}
				}
			}
		}
	}
	
	public void fullConnectPoolMap() {
		double alpha = 1;
		int i=0;
		for(int n=0;n<poolNum;n++) {
			for(int h=0;h<poolMapHeight;h++) {
				for(int w=0;w<poolMapWidth;w++) {
					fullConnect[i] = alpha * poolMap[n][h][w];
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
	}
	
	public static void main(String args[]) throws IOException {
		CNNSingleLayerFormatInput cnn = new CNNSingleLayerFormatInput();
		
		cnn.randomConvCore();
		cnn.initPoolCore();
		cnn.initFullConnectToUnitWeight();
		cnn.randomUnitToOutputWeight();
		
		System.out.println(cnn.fullNum);
		
		int circle = 10000;
//		double avgPixel = cnn.getAveragePixel();
		for(int c=0;c<circle;c++) {
			for(int j=0;j<1000;j++) {
				String filePath = "E:\\Dataset(A feature-hybrid malware variants detection )\\benign-png\\"+j+".png";
				cnn.initMatrixAndValue(filePath,0);
				cnn.convolutionMatrix();
				cnn.poolingConvMap();
				cnn.fullConnectPoolMap();
				cnn.softmaxFullConnect(0);
				cnn.backPropagation();
			}
			for(int j=0;j<1000;j++) {
				String filePath = "E:\\Dataset(A feature-hybrid malware variants detection )\\malicious-png\\"+j+".png";
				cnn.initMatrixAndValue(filePath,1);
				cnn.convolutionMatrix();
				cnn.poolingConvMap();
				cnn.fullConnectPoolMap();
				cnn.softmaxFullConnect(1);
				cnn.backPropagation();
			}
			FileWriter fw1 = null;
			FileWriter fw2 = null;
			try {
				File f1=new File("E:\\Dataset(A feature-hybrid malware variants detection )\\malwareOpcodeFeature.txt");
				f1.delete();
				fw1 = new FileWriter(f1, true);
				File f2=new File("E:\\Dataset(A feature-hybrid malware variants detection )\\benignOpcodeFeature.txt");
				f2.delete();
				fw2 = new FileWriter(f2, true);
			} catch (Exception e) {
				e.printStackTrace();
			}
			PrintWriter pw1 = new PrintWriter(fw1);
			PrintWriter pw2 = new PrintWriter(fw2);
			int right = 0;
			int wrong = 0;
			for(int j=0;j<1000;j++) {
				String filePath = "E:\\Dataset(A feature-hybrid malware variants detection )\\malicious-png\\"+j+".png";
				cnn.initMatrixAndValue(filePath,0);
				cnn.convolutionMatrix();
				cnn.poolingConvMap();
				cnn.fullConnectPoolMap();
				int rs = cnn.softmaxFullConnect(0);
				for(int k=0;k<cnn.unitNum;k++) {
					pw1.print(cnn.unit[k]+" ");
				}
				pw1.println();
				pw1.flush();
				if(rs==1) {
					right++;
				}else {
					wrong++;
				}
			}
			pw1.close();
			for(int j=0;j<1000;j++) {
				String filePath = "E:\\Dataset(A feature-hybrid malware variants detection )\\benign-png\\"+j+".png";
				cnn.initMatrixAndValue(filePath,1);
				cnn.convolutionMatrix();
				cnn.poolingConvMap();
				cnn.fullConnectPoolMap();
				int rs = cnn.softmaxFullConnect(1);
				for(int k=0;k<cnn.unitNum;k++) {
					pw2.print(cnn.unit[k]+" ");
				}
				pw2.println();
				pw2.flush();
				if(rs==1) {
					right++;
				}else {
					wrong++;
				}
			}
			pw2.close();
			System.out.println(c+"-----------------------------------------");
			System.out.println(right+" "+wrong);
			double accuracy = (double)right * 100 / (right+wrong);
			System.out.println("accuracy: "+accuracy+" %");
		}
	}

}
