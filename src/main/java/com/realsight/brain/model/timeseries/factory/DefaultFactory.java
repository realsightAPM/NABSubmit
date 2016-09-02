package com.realsight.brain.model.timeseries.factory;

import com.realsight.brain.model.timeseries.regression.LinearRegression;

public class DefaultFactory extends AbstractFactory{

	@Override
	public LinearRegression createLinearRegression() {
		// TODO Auto-generated method stub
		return new LinearRegression();
	}

}
