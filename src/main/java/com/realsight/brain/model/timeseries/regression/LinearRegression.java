package com.realsight.brain.model.timeseries.regression;

import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import com.realsight.brain.model.timeseries.regression.basic.BasicModel;

public class LinearRegression implements BasicModel {
	
	private double mn = Double.MAX_VALUE;
	private double mx = Double.MIN_VALUE;
	private final double eps = 1e-8;
	
	private double[][] inputs = null;
	private double[] outputs = null;
	private DoubleMatrix delta = null;
	
	private double transformTime(double x){
		if(Math.abs(mx - mn) < eps)
			return 0;
		return (x-mn)/(mx-mn);
	}
	
	private double[][] getTrainingInput(List<Double> X, final List<Double>[] features){
		List<Double> t = new ArrayList<Double>();
		for(int i = 0; i < X.size(); i++){
			mn = Math.min(X.get(i), mn);
			mx = Math.max(X.get(i), mx);
		}
		for(int i = 0; i < X.size(); i++){
			double x = X.get(i);
			t.add(transformTime(x));
		}
		double[][] inputs = new double[t.size()][];
		for(int i = 0; i < t.size(); i++){
			inputs[i] = new double[5+features.length];
			inputs[i][0] = 1;
			inputs[i][1] = t.get(i);
			inputs[i][2] = Math.pow(t.get(i), 2); 	//Math.pow(t.get(i), 2);
			inputs[i][3] = Math.sqrt(t.get(i)); 	//Math.sqrt(t.get(i));
			inputs[i][4] = Math.pow(t.get(i), 3); 	//Math.pow(t.get(i), 3);
			for(int j = 0; j < features.length; j++){
				inputs[i][5+j] = features[j].get(i);
			}
		}
		return inputs;
	}
	
	private double[] getTrainingOutput(List<Double> Y){
		double[] output = new double[Y.size()];
		for(int i = 0; i < Y.size(); i++){
			output[i] = Y.get(i);
		}
		return output;
	}
	
	public void training(List<Double> t, List<Double> s, final List<Double>[] features){
		inputs = getTrainingInput(t, features);
		outputs = getTrainingOutput(s);	
		DoubleMatrix Y = new DoubleMatrix(outputs);
		DoubleMatrix X = new DoubleMatrix(inputs);
		DoubleMatrix Xt = X.transpose();
		DoubleMatrix XtX = Xt.mmul(X);
		DoubleMatrix invX = Solve.pinv(XtX);
		this.delta = invX.mmul(Xt).mmul(Y).transpose();
		//System.out.println(this.delta);
	}
	
	public void training(List<Double> s, final List<Double>[] features){
		List<Double> t = new ArrayList<Double>();
		for(int i = 0; i < s.size(); i++){
			t.add(0.0);
		}
		training(t, s, features);
	}
	
	public void training(List<Double> t, List<Double> s){
		@SuppressWarnings("unchecked")
		final List<Double>[] features = (List<Double>[]) new List<?>[0];
		training(t, s, features);
	}
	
	public double forecasting(double fx, double[] feature){
		fx = transformTime(fx);
		DoubleMatrix x = new DoubleMatrix(5+feature.length);
		x.put(new int[]{0}, 1);
		x.put(new int[]{1}, fx);
		x.put(new int[]{2}, Math.pow(fx, 2)/*Math.pow(fx, 2)*/);
		x.put(new int[]{3}, Math.sqrt(fx)/*Math.sqrt(fx)*/);
		x.put(new int[]{4}, Math.pow(fx, 3)/*Math.pow(fx, 3)*/);
		for(int i = 0; i < feature.length; i++){
			x.put(new int[]{5+i}, feature[i]);
		}
		if(delta == null)
			return 0;
		if(delta.getColumns() != x.getRows())
			return 0;
		//System.out.println("delta = " + delta);
		//System.out.println("x = " + x);
		return this.delta.mmul(x).get(0, 0);
	}
	
	public double forecasting(double fx){
		final double[] features = new double[0];
		return forecasting(fx, features);
	}
	
	public double forecasting(double[] features){
		final double fx = 0;
		return forecasting(fx, features);
	}
	
	public double SSE(){
		if(this.delta == null)
			return 0;
		DoubleMatrix Y = new DoubleMatrix(outputs);
		DoubleMatrix X = new DoubleMatrix(inputs);
		return Math.pow(Y.transpose().distance2(delta.mmul(X.transpose())), 2);
	}
	
	public double NMSE(){
		if(this.delta == null)
			return 0;
		DoubleMatrix Y = new DoubleMatrix(outputs);
		DoubleMatrix X = new DoubleMatrix(inputs);
		return Math.pow(Y.transpose().sub(delta.mmul(X.transpose())).norm2(), 1)/outputs.length+1e-6;
	}
	
	public double SD(){
		if(this.delta == null)
			return 0;
		DoubleMatrix Y = new DoubleMatrix(outputs);
		DoubleMatrix X = new DoubleMatrix(inputs);
		return Y.transpose().distance2(delta.mmul(X.transpose())) + 1e-6;
	}
}
