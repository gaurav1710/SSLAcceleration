package com.sslproxy.benchmark;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataCollector {
	private static List<DataPoint> dataPoints;

	public static synchronized void addPoint(DataPoint dataPoint) {
		dataPoints.add(dataPoint);
	}

	public static void reset() {
		dataPoints = new ArrayList<DataPoint>();
	}

	public static List<DataPoint> getDataPoints() {
		return dataPoints;
	}

	public static void setDataPoints(List<DataPoint> dataPoints) {
		DataCollector.dataPoints = dataPoints;
	}

	public static void saveInFile(String filePath) {
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(new File(filePath)));
			for (DataPoint dp : dataPoints) {
				bw.append(dp.getX() + "," + dp.getY() + "\n");
			}
			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}