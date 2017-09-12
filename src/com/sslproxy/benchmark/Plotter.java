package com.sslproxy.benchmark;

import java.awt.BasicStroke;
import java.awt.Color;
import java.io.File;
import java.io.IOException;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class Plotter {
	public static void plot(String filePath, String title, String yLegend, String xLegend) {

		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries gpuSeries = new XYSeries("GPU");
		for (DataPoint dp : DataCollector.getDataPoints()) {
			// dataset.setValue(new Integer(((int)dp.getX())), "GPU",
			// ""+dp.getY());
			gpuSeries.add(new Integer((int) dp.getX()), new Double(dp.getY()));
		}

		dataset.addSeries(gpuSeries);

		JFreeChart lineChart = ChartFactory.createXYLineChart(title, xLegend, yLegend, dataset,
				PlotOrientation.VERTICAL, true, false, false);
		lineChart.getPlot().setBackgroundPaint(Color.LIGHT_GRAY);

		((XYPlot) lineChart.getPlot()).getRenderer().setBaseStroke(new BasicStroke(3));
		((XYPlot) lineChart.getPlot()).getRenderer().setSeriesStroke(0, new BasicStroke(3));
		;

		int width = 1900;
		int height = 640;
		File barChartFile = new File(filePath);

		try {
			ChartUtilities.saveChartAsJPEG(barChartFile, lineChart, width, height);
		} catch (IOException e) {
			System.out.println(e);
		}
	}
}