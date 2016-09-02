package com.realsight.brain.model.timeseries.regression.basic;

import java.util.List;

public interface BasicModel {
	public void training(List<Double> t, List<Double> s, final List<Double>[] features);
	public void training(List<Double> s, final List<Double>[] features);
	public void training(List<Double> t, List<Double> s);
	public double forecasting(double fx, double[] feature);
	public double forecasting(double fx);
	public double forecasting(double[] features);
}
