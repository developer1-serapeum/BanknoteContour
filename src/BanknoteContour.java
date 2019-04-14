import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Point;

public class BanknoteContour {

	public static void main(String[] args){

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		File folder = new File("images");
		File[] listOfFiles = folder.listFiles();

		String fileName = "";

		// Loop over all the test images
		for (int i = 0; i < listOfFiles.length; i++) {

			fileName = listOfFiles[i].getName();

			System.out.println("Reading image : " + fileName);

			// Read the image
			Mat src = Imgcodecs.imread(folder + "/"+ fileName, Imgcodecs.IMREAD_COLOR);

			// Reduce noise by filtering
			Mat srcFiltered = new Mat();
			Size ksize = new Size(5, 5);
			double sigmaX = 0;
			double sigmaY = 0;
			Imgproc.GaussianBlur(src, srcFiltered, ksize, sigmaX, sigmaY , Core.BORDER_DEFAULT);

			// Run the k-means clustering algorithm
			List<Mat> clusters = cluster(src, 5);

			// Draw the largest cluster
			Imgcodecs.imwrite("result/"+fileName, clusters.get(0));

		}

		System.out.println("Finished finding contours ..." );
		System.exit(0);        

	}

	/**
	 * Draw a rotated rectangle over a given drawing matrix.
	 * @param drawing: The matrix the hold the image
	 * @param vertices: The four corners of the rectangle
	 * @param color: The color used to draw the rectangle lines
	 */
	private static void drawRotatedRect(Mat drawing, Point[] vertices, Scalar color) {

		for( int k = 0; k < 4; k++ ) {
			Imgproc.line( drawing, vertices[k], vertices[(k+1)%4], color, Imgproc.LINE_8, Imgproc.LINE_8 );
		}
	}


	public static List<Mat> cluster(Mat cutout, int k) {

		Mat samples = cutout.reshape(1, cutout.cols() * cutout.rows());
		Mat samples32f = new Mat();
		samples.convertTo(samples32f, CvType.CV_32F, 1.0 / 255.0);

		Mat labels = new Mat();
		TermCriteria criteria = new TermCriteria(TermCriteria.COUNT, 100, 1);
		Mat centers = new Mat();
		
		Core.kmeans(samples32f, k, labels, criteria, 1, Core.KMEANS_PP_CENTERS, centers);

		return showClusters(cutout, labels, centers);
	}

	private static List<Mat> showClusters (Mat cutout, Mat labels, Mat centers) {

		centers.convertTo(centers, CvType.CV_8UC1, 255.0);

		centers.reshape(3);

		List<Mat> clusters = new ArrayList<Mat>();

		for(int i = 0; i < centers.rows(); i++) {
			clusters.add(Mat.zeros(cutout.size(), cutout.type()));
		}

		Map<Integer, Integer> counts = new HashMap<Integer, Integer>();

		for(int i = 0; i < centers.rows(); i++) counts.put(i, 0);

		int rows = 0;

		for(int y = 0; y < cutout.rows(); y++) {

			for(int x = 0; x < cutout.cols(); x++) {

				int label = (int)labels.get(rows, 0)[0];

				int r = (int)centers.get(label, 2)[0];
				int g = (int)centers.get(label, 1)[0];
				int b = (int)centers.get(label, 0)[0];

				clusters.get(label).put(y, x, b, g, r);

				rows++;
			}
		}

		return clusters;
	}
	
	private Mat segmentByColor(Mat imgRgb) {
        
		Mat srcHsv = new Mat();
        Imgproc.cvtColor(imgRgb, srcHsv, Imgproc.COLOR_BGR2HSV);

        // Range for lower color
        Mat lowerColor = new Mat();
        Scalar lower = new Scalar(0, 10, 150);
        Scalar upper = new Scalar(40, 255, 200);
        Core.inRange(srcHsv, lower, upper, lowerColor);
        
        // Range for upper color
        Mat upperColor = new Mat();
        lower = new Scalar(160, 10, 10);
        upper = new Scalar(180, 255, 255);
        Core.inRange(srcHsv, lower, upper, upperColor);
        
        Mat mainColor = new Mat();
        Core.add(lowerColor, upperColor, mainColor);
        
        return mainColor;
	}
}
