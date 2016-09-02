package com.realsight.brain.model.noiseseries;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jblas.DoubleMatrix;

import com.realsight.brain.model.timeseries.factory.DefaultFactory;
import com.realsight.brain.model.timeseries.regression.basic.BasicModel;

/* @author sunmuxin
 * 
 */

public class ChangePoint {
	private static final Logger logger = LogManager.getLogger(ChangePoint.class.getName());
	// dp_likelihood_value[i][j] show: In front of i-th time-series existed j change-points, likelihood value.
	private static double[][] dp_penalty_value = null;
	// Minimum distance between change-points.
	private static int min_dist = -1;
	private static final double eps = 1e-8;
	private static List<Double> e = null;
	private static double[] sse = null;
	
	public static void training(List<Double> noiseseries){
		ChangePoint.e = noiseseries;
		final int n = e.size();
		min_dist = (int) n/40;
		dp_penalty_value = new double[n+1][];
		sse = new double[n+1];
		sse[0] = 0;
		dp_penalty_value[0] = new double[n/min_dist+1];
		Arrays.fill(dp_penalty_value[0], Double.MAX_VALUE);
		dp_penalty_value[0][0] = 0;
		logger.info("train start ...");
		for(int i = 1; i <= n; i++){
			if(i % 500 == 0)
				logger.info("train completed " + (100.0*i/n) + "% ...");
			sse[i] = sse[i-1] + Math.pow(e.get(i-1), 2);
			dp_penalty_value[i] = new double[n/min_dist+1];
			Arrays.fill(dp_penalty_value[i], Double.MAX_VALUE);
			for(int j = 1; j*min_dist < i; j++){
				for(int k = 0; i-k > min_dist; k++){
					//logger.info("dp_penalty_value["+i+","+k+"]="+((sse[i]-sse[k])/(i-k)+eps));
					dp_penalty_value[i][j] = Math.min(dp_penalty_value[i][j], 
							dp_penalty_value[k][j-1] + (i-k)*Math.log((sse[i]-sse[k])/(i-k)+eps));
				}
				//logger.info("dp_penalty_value["+i+","+j+"]="+dp_penalty_value[i][j]);
			}
		}
		logger.info("train end ...");
	}
	
	private static  List<Integer> getChangePoints(int i, int j){
		List<Integer> res = null;
		if(i==0 && j==0)
			return new ArrayList<Integer>();
		else if(i==0){
			logger.error("dp_penalty_value is error. " +
					"please check all noise series change-point code. i error");
			throw new RuntimeException("dp_penalty_value is error. " +
					"please check all noise series change-point code. i error");
		}
		else if(j==0){
			logger.error("dp_penalty_value is error. " +
					"please check all noise series change-point code. j error");
			throw new RuntimeException("dp_penalty_value is error. " +
					"please check all noise series change-point code. j error");
		}
		for(int k = 0; i-k > min_dist; k++){
			double penalty_value = dp_penalty_value[k][j-1] + (i-k)*Math.log((sse[i]-sse[k])/(i-k)+eps);
			if(Math.abs(penalty_value-dp_penalty_value[i][j]) < eps){
				res = getChangePoints(k, j-1);
				res.add(k);
				return res;
			}
		}
		logger.error("@author sunmuxin also don't know what happened.");
		throw new RuntimeException("@author sunmuxin also don't know what happened.");
	}
	
	public static List<Integer> getChangePoints(int num) throws IndexOutOfBoundsException{
		if(e == null){
			throw new NullPointerException("no exited noise serires.");
		}
		final int n = e.size();
		if(num+1>n/min_dist || dp_penalty_value[n][num+1]==Double.MAX_VALUE){
			throw new IndexOutOfBoundsException("no exited " + num + " change-points");
		}
		return getChangePoints(n, num+1);
	}
	
	public static double getPenaltyValue(int num) throws IndexOutOfBoundsException{
		if(e == null){
			throw new NullPointerException("no exited noise serires.");
		}
		final int n = e.size();
		if(num+1>n/min_dist){
			throw new IndexOutOfBoundsException("no exited " + num + " change-points");
		}
		return dp_penalty_value[n][num+1];
	}
	
	private static final DefaultFactory df = new DefaultFactory();
	private static final BasicModel bm = df.createLinearRegression();
	private static final int feature_length = 5;
	
	public static List<Double> detectorNoiseSeriesAnomaly(List<Double> s){
		
		int n = s.size();
		
		@SuppressWarnings("unchecked")
		List<Double>[] features = (List<Double>[]) new List<?>[feature_length];
		for(int i = 0; i < feature_length; i++){
			features[i] = s.subList(i, n-feature_length+i);
		}
		bm.training(s.subList(feature_length, n), features);
		
		DoubleMatrix Y = new DoubleMatrix(s.size());
		DoubleMatrix _Y = new DoubleMatrix(s.size());
		List<Double> noiseseries = new ArrayList<Double>();
		double var = 0.0;
		for(int i = 0; i < s.size(); i++){
			Y.put(i, s.get(i));
			if(i < feature_length){
				_Y.put(i, s.get(i));
				noiseseries.add(0.0);
			} else{
				double[] feature = new double[feature_length];
				for(int j = 0; j < feature_length; j++){
					feature[j] = s.get(i-feature_length+j);
				}
				_Y.put(i, bm.forecasting(feature));
				noiseseries.add(Y.get(i)-bm.forecasting(feature));
			}
			var += noiseseries.get(i) * noiseseries.get(i);
		}
		double sd = Math.sqrt(var/n);
		double sum = 0.0;
		List<Double> res = new ArrayList<Double>();
		for(int i = 0; i < s.size(); i++){
			sum += Math.abs(noiseseries.get(i)/sd);
			res.add(0.0);
		}
		double avr = sum/n;
		
		training(noiseseries);
		
		int j = 6;
		List<Integer> cp = null;
		try{
			cp = getChangePoints(j);
			cp.add(n-1);
		} catch (IndexOutOfBoundsException e){
			logger.info("change-point error.");
		} catch (NullPointerException e){
			logger.error(e.getMessage());
		}
		for(int k = 1; k < cp.size()-1; k += 1){
			int cp_id = cp.get(k);
			res.set(cp_id, avr);
		}
		return res;
	}
	
	public static void main(String[] args) throws IOException{
		File root = new File("D:/workspace/NAB/src/results/null/");
		for(File file : root.listFiles()){
			if(file.isDirectory()){
				for(File dir : file.listFiles()){
					if(dir.isDirectory())
						continue;
					List<String> t = new ArrayList<String>();
					List<Double> s = new ArrayList<Double>();
					List<String> l = new ArrayList<String>();
					List<String> FP = new ArrayList<String>();
					List<String> FN = new ArrayList<String>();
					List<String> S = new ArrayList<String>();
					Scanner sin = new Scanner(dir);
					sin.nextLine();
					while(sin.hasNext()){
						String line = sin.nextLine();
						t.add(line.split(",")[0]);
						s.add(Double.parseDouble(line.split(",")[1]));
						l.add(line.split(",")[3]);
						FP.add(line.split(",")[4]);
						FN.add(line.split(",")[5]);
						S.add(line.split(",")[6]);
					}
					logger.info(dir.getPath());
					List<Double> a = detectorNoiseSeriesAnomaly(s);
					String resultFileName = dir.getPath().replace("null", "neusoft");
					String resultPath = new File(resultFileName).getParent();
					if(!new File(resultPath).exists()){
						new File(resultPath).mkdirs();
					}
					
					OutputStream os = new FileOutputStream(resultFileName);
			        OutputStreamWriter writer = new OutputStreamWriter(os);
			        writer.write("timestamp,value,anomaly_score,label,S(t)_reward_low_FP_rate,S(t)_reward_low_FN_rate,S(t)_standard\n");
			        for(int i = 0; i < s.size(); i++){
			        	writer.write(t.get(i)+","+s.get(i)+","+a.get(i)+","+l.get(i)+","+FP.get(i)+","+FN.get(i)+","+S.get(i)+"\n");
			        }
			        writer.close();
				}
			}
		}
		
	}
}
