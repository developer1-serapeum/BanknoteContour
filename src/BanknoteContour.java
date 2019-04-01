import java.io.File;
import java.util.ArrayList;
import java.util.List;

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

            Mat src = Imgcodecs.imread(folder + "/"+ fileName, Imgcodecs.IMREAD_GRAYSCALE);
            Mat srcColor = Imgcodecs.imread(folder + "/"+ fileName, Imgcodecs.IMREAD_COLOR);
           
            
            // Reduce noise by filtering
            Mat srcBlurred = new Mat();            
            Size ksize = new Size(5, 5);
            double sigmaX = 0;
            double sigmaY = 0;
            Imgproc.GaussianBlur(src, srcBlurred, ksize, sigmaX, sigmaY , Core.BORDER_DEFAULT);
            
            
            // Get the strong corners in the source image
            MatOfPoint corners = new MatOfPoint();
            int maxCorners = 20;
            double qualityLevel = 0.1;
            double minDistance = 10;
            Imgproc.goodFeaturesToTrack(srcBlurred, corners, maxCorners, qualityLevel, minDistance);
            
            // Draw corners detected
            Mat drawing = src;
            int radius = 8;
            List<Point> points = corners.toList();
            for (int j=0; j < corners.rows(); j++) {
                Imgproc.circle(drawing, points.get(j), radius, new Scalar(0, 255, 0), Imgproc.LINE_8);
            }
            
            
            /*
            // Sharpen the image to emphasize the edges
            Mat kernelSharpening = new Mat( new Size(3,3), CvType.CV_32F);
            double[] data = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
            kernelSharpening.put(0, 0, data);
            Mat srcSharpened = new Mat();
            Imgproc.filter2D(src, srcSharpened, -1, kernelSharpening);

            // Improve the contrast to account for lighting conditions
            Mat srcEqualized = new Mat();
            Imgproc.equalizeHist(srcBlurred, srcEqualized);
            
            // Get rid of self intersecting contours
            Mat cleaned = new Mat();
            int kernelSize = 2;
            int elementType = Imgproc.CV_SHAPE_RECT;
            Mat element = Imgproc.getStructuringElement(
                    elementType, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                    new Point(kernelSize, kernelSize));
            Imgproc.dilate(srcEqualized, cleaned, element, new Point(element.width()/2, element.height()/2), 4);
            //Imgproc.erode(src, src, element, new Point(element.width()/2, element.height()/2), 1);
                       
            // threshold to find the contour
            Mat thresholded = new Mat();
            Imgproc.threshold(cleaned, thresholded, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            // Find edges
            Mat cannyOutput = new Mat();
            int cannyThreshold = 70;
            Imgproc.Canny(srcBlurred, cannyOutput, cannyThreshold, cannyThreshold * 2);

            // Increase contour thickness
            Mat dilated  = new Mat();
            Imgproc.dilate(cannyOutput, dilated, element, new Point(element.width()/2, element.height()/2), 4);

            //Mat drawing = cannyOutput;
            //Mat drawing  = new Mat(src.size(), CvType.CV_8UC3);
            Mat drawing = src;

            // Find all contours
            Mat hierarchy = new Mat();
            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(cannyOutput, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            //System.out.println("Number of contours = " + String.valueOf(contours.size()));

            Scalar green = new Scalar(0, 255, 0);
            Scalar blue = new Scalar(0, 0, 255);

            // Structural shape analysis
            double contourArea = 0.0;
            double contourLength = 0.0;
            double minContourLength = 0.0;
            MatOfPoint contour = new MatOfPoint();
            MatOfPoint2f contour2f = new MatOfPoint2f();

            RotatedRect rect = new RotatedRect();
            Point[] vertices = new Point[4];
            MatOfPoint verticesMat = new MatOfPoint();

            for (int j = 0; j < contours.size(); j++) {

                contour = contours.get(j);
                contour.convertTo(contour2f, CvType.CV_32F);

                //contourArea = Imgproc.contourArea(contours.get(i));
                contourLength = Imgproc.arcLength(contour2f, false);
                minContourLength = 0.9 * 1 * src.width() + 0.9 * 1 * src.height();

                if (contourLength > minContourLength ) {

                    rect = Imgproc.minAreaRect(contour2f);
                    rect.points(vertices);
                    verticesMat.fromArray(vertices);

                    Imgproc.drawContours(drawing, contours, j, green, Imgproc.LINE_8, Imgproc.LINE_8, hierarchy, 0);
                    
                    drawRotatedRect(drawing, vertices, blue);
                }
            }
            */
            
            
            Imgcodecs.imwrite("result/"+fileName, drawing);
        
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
