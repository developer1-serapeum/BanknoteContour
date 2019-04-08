import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Point;

public class BanknoteContour {

    public static void main(String[] args){

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Loop over all the test images
        File folder = new File("images");
        File[] listOfFiles = folder.listFiles();
        String fileName = "";

        for (int i = 0; i < listOfFiles.length; i++) {

            fileName = listOfFiles[i].getName();

            System.out.println("Reading image : " + fileName);

            Mat src = Imgcodecs.imread(folder + "/"+ fileName, Imgcodecs.IMREAD_COLOR);
            
            // Create a kernel that we will use to sharpen our image
            Mat kernel = new Mat(3, 3, CvType.CV_32F);
            // an approximation of second derivative, a quite strong kernel
            float[] kernelData = new float[(int) (kernel.total() * kernel.channels())];
            kernelData[0] = 1; kernelData[1] = 1; kernelData[2] = 1;
            kernelData[3] = 1; kernelData[4] = -8; kernelData[5] = 1;
            kernelData[6] = 1; kernelData[7] = 1; kernelData[8] = 1;
            kernel.put(0, 0, kernelData);

            // Do the laplacian filtering and convert to CV_32
            // in order not to truncate negative numbers
            Mat imgLaplacian = new Mat();
            Imgproc.filter2D(src, imgLaplacian, CvType.CV_32F, kernel);
            Mat sharp = new Mat();
            src.convertTo(sharp, CvType.CV_32F);
            Mat imgSharpened = new Mat();
            Core.subtract(sharp, imgLaplacian, imgSharpened);

            // convert back to 8bits gray scale
            imgSharpened.convertTo(imgSharpened, CvType.CV_8UC3);
            imgLaplacian.convertTo(imgLaplacian, CvType.CV_8UC3);

            // Create binary image from source image
            Mat bw = new Mat();
            Imgproc.cvtColor(imgSharpened, bw, Imgproc.COLOR_BGR2GRAY);
            Imgproc.threshold(bw, bw, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
            

            // Perform the distance transform algorithm
            Mat dist = new Mat();
            Imgproc.distanceTransform(bw, dist, Imgproc.DIST_C, 3);

            // Normalize the distance image for range = {0.0, 1.0}
            // so we can visualize and threshold it
            Mat dist2 = new Mat();
            Core.normalize(dist, dist2, 0.0, 1.0, Core.NORM_MINMAX);
            Mat distDisplayScaled = new Mat();
            Core.multiply(dist2, new Scalar(255), distDisplayScaled);
            Mat distDisplay = new Mat();
            distDisplayScaled.convertTo(distDisplay, CvType.CV_8U);
            

            // Threshold to obtain the peaks
            // This will be the markers for the foreground objects
            Mat dstThreshold = new Mat();
            Imgproc.threshold(dist, dstThreshold, 0.4, 1.0, Imgproc.THRESH_BINARY);

            // Dilate a bit the dist image
            Mat kernel1 = Mat.ones(3, 3, CvType.CV_8U);
            Imgproc.dilate(dstThreshold, dstThreshold, kernel1);
            Mat distDisplay2 = new Mat();
            dist.convertTo(distDisplay2, CvType.CV_8U);
            Core.multiply(distDisplay2, new Scalar(255), distDisplay2);

            // Create the CV_8U version of the distance image
            // It is needed for findContours()
            Mat dist_8u = new Mat();
            dist.convertTo(dist_8u, CvType.CV_8U);

            // Find total markers
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(dist_8u, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // Create the marker image for the watershed algorithm
            Mat markers = Mat.zeros(dist.size(), CvType.CV_32S);

            
            // Draw the foreground markers
            for (int j = 0; j < contours.size(); j++) {
                Imgproc.drawContours(markers, contours, j, new Scalar(j + 1), -1);
            }
            
            

            // Draw the background marker
            Mat markersScaled = new Mat();
            markers.convertTo(markersScaled, CvType.CV_32F);
            Core.normalize(markersScaled, markersScaled, 0.0, 255.0, Core.NORM_MINMAX);
            Imgproc.circle(markersScaled, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
            Mat markersDisplay = new Mat();
            markersScaled.convertTo(markersDisplay, CvType.CV_8U);
            
            Imgproc.circle(markers, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);

            // Perform the watershed algorithm
            Imgproc.watershed(imgSharpened, markers);

            Mat mark = Mat.zeros(markers.size(), CvType.CV_8U);
            markers.convertTo(mark, CvType.CV_8UC1);
            Core.bitwise_not(mark, mark);
            
            // Generate random colors
            Random rng = new Random(12345);
            List<Scalar> colors = new ArrayList<>(contours.size());
            for (int i1 = 0; i1 < contours.size(); i1++) {
                int b = rng.nextInt(256);
                int g = rng.nextInt(256);
                int r = rng.nextInt(256);

                colors.add(new Scalar(b, g, r));
            }

            // Create the result image
            Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC3);
            byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
            dst.get(0, 0, dstData);

            // Fill labeled objects with random colors
            int[] markersData = new int[(int) (markers.total() * markers.channels())];
            markers.get(0, 0, markersData);
            
            /*
            for (int k = 0; k < markers.rows(); i++) {
                for (int j = 0; j < markers.cols(); j++) {
                    int index = markersData[k * markers.cols() + j];
                    if (index > 0 && index <= contours.size()) {
                        dstData[(k * dst.cols() + j) * 3 + 0] = (byte) colors.get(index - 1).val[0];
                        dstData[(k * dst.cols() + j) * 3 + 1] = (byte) colors.get(index - 1).val[1];
                        dstData[(k * dst.cols() + j) * 3 + 2] = (byte) colors.get(index - 1).val[2];
                    } else {
                        dstData[(k * dst.cols() + j) * 3 + 0] = 0;
                        dstData[(k * dst.cols() + j) * 3 + 1] = 0;
                        dstData[(k * dst.cols() + j) * 3 + 2] = 0;
                    }
                }
            }
            */
            dst.put(0, 0, dstData);
            
            
            Imgcodecs.imwrite("result/"+fileName, dist_8u);
        
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
}
