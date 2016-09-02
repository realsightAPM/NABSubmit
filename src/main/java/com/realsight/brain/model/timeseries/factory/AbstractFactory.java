package com.realsight.brain.model.timeseries.factory;

import com.realsight.brain.model.timeseries.regression.LinearRegression;

public abstract class AbstractFactory {
	public abstract LinearRegression createLinearRegression();
}
